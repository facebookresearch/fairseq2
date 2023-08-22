// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "fairseq2n/data/text/sentencepiece/sp_processor.h"

#include <stdexcept>
#include <system_error>

#include <fmt/format.h>
#include <sentencepiece/src/builtin_pb/sentencepiece_model.pb.h>

#include "fairseq2n/detail/exception.h"
#include "fairseq2n/utils/cast.h"

using sentencepiece::ImmutableSentencePieceText;
using sentencepiece::ModelProto;
using sentencepiece::ModelProto_SentencePiece;
using sentencepiece::SentencePieceProcessor;

namespace fairseq2n::detail {

class sp_model_proto_loader {
public:
    explicit
    sp_model_proto_loader(std::string_view pathname, sp_model_options &&opts) noexcept
      : pathname_{pathname}, opts_{std::move(opts)}
    {}

    std::unique_ptr<ModelProto> &&
    load() &&;

private:
    void
    load_proto();

    void
    add_control_symbols();

    void
    add_piece(std::string &&piece);

private:
    std::string_view pathname_;
    sp_model_options opts_;
    std::unique_ptr<ModelProto> proto_{};
    std::unique_ptr<SentencePieceProcessor> processor_{};
};

std::unique_ptr<ModelProto> &&
sp_model_proto_loader::load() &&
{
    load_proto();

    add_control_symbols();

    return std::move(proto_);
}

void
sp_model_proto_loader::load_proto()
{
    proto_ = std::make_unique<ModelProto>();

    auto st = sentencepiece::io::LoadModelProto(pathname_, proto_.get());
    if (st.ok())
        return;

    if (st.code() == sentencepiece::util::StatusCode::kNotFound)
        throw_system_error(std::make_error_code(std::errc::no_such_file_or_directory),
            "The SentencePiece model '{}' cannot be opened", pathname_);

    if (st.code() == sentencepiece::util::StatusCode::kPermissionDenied)
        throw_system_error(std::make_error_code(std::errc::permission_denied),
            "The SentencePiece model '{}' cannot be opened", pathname_);

    throw_<std::runtime_error>(
        "The SentecePiece model '{}' cannot be opened. {}", pathname_, st.message());
}

void
sp_model_proto_loader::add_control_symbols()
{
    for (std::string &symbol : opts_.control_symbols()) {
        if (symbol.empty())
            continue;

        if (symbol == "<pad>" || symbol == "<pad>@0") {
            proto_->mutable_trainer_spec()->set_pad_piece("<pad>");

            add_piece("<pad>");

            // This is a workaround for SentencePiece models that, for legacy
            // reasons, do not have a pad symbol, but expected to have one at
            // index 0 (e.g. NLLB models).
            if (symbol == "<pad>@0") {
                auto *pieces = proto_->mutable_pieces();

                // `RepeatedPtrField` does not have an insert method, so we move
                // our pad symbol from the end to the beginning of the list.
                for (int i = pieces->size() - 1; i > 0; --i)
                    pieces->SwapElements(i, i - 1);
            }
        } else
            add_piece(std::move(symbol));
    }
}

void
sp_model_proto_loader::add_piece(std::string &&piece)
{
    ModelProto_SentencePiece *sp = proto_->add_pieces();

    sp->set_piece(std::move(piece));

    sp->set_type(ModelProto_SentencePiece::CONTROL);
}

std::unique_ptr<sp_processor>
sp_processor::from_serialized(std::string_view serialized)
{
    auto proto = std::make_unique<ModelProto>();

    bool r = proto->ParseFromArray(serialized.data(), static_cast<int>(serialized.size()));
    if (!r)
        throw_<std::invalid_argument>(
            "`serialized` must be a serialized Protobuf object, but cannot be parsed as such.");

    sp_processor processor{std::move(proto)};

    return std::make_unique<sp_processor>(std::move(processor));
}

sp_processor::sp_processor(std::unique_ptr<ModelProto> &&proto)
{
    native_ = std::make_unique<SentencePieceProcessor>();

    auto st = native_->Load(std::move(proto));
    if (!st.ok())
        throw_<std::runtime_error>(st.message());

    unk_idx = conditional_cast<std::int32_t>(native_->unk_id());
    bos_idx = conditional_cast<std::int32_t>(native_->bos_id());
    eos_idx = conditional_cast<std::int32_t>(native_->eos_id());
    pad_idx = conditional_cast<std::int32_t>(native_->pad_id());

    vocabulary_size = conditional_cast<std::size_t>(native_->GetPieceSize());
}

sp_processor::sp_processor(std::string_view model_pathname, sp_model_options &&opts)
  : sp_processor{sp_model_proto_loader{model_pathname, std::move(opts)}.load()}
{}

ImmutableSentencePieceText
sp_processor::encode(std::string_view text) const
{
    ImmutableSentencePieceText spt{};

    auto st = native_->Encode(text, spt.mutable_proto());
    if (!st.ok())
        throw_<std::runtime_error>(st.message());

    return spt;
}

ImmutableSentencePieceText
sp_processor::sample(std::string_view text, std::int32_t nbest_size, float alpha) const
{
    ImmutableSentencePieceText spt{};

    auto st = native_->SampleEncode(text, nbest_size, alpha, spt.mutable_proto());
    if (!st.ok())
        throw_<std::runtime_error>(st.message());

    return spt;
}

std::string
sp_processor::decode(const std::vector<std::string_view> &tokens) const
{
    std::string text{};

    auto st = native_->Decode(tokens, &text);
    if (!st.ok())
        throw_<std::runtime_error>(st.message());

    return text;
}

std::int32_t
sp_processor::token_to_index(std::string_view token) const
{
    return conditional_cast<std::int32_t>(native_->PieceToId(token));
}

std::string_view
sp_processor::index_to_token(std::int32_t idx) const
{
    if (static_cast<std::size_t>(idx) >= vocabulary_size)
        throw_<std::invalid_argument>(
            "`idx` must be less than vocabulary size ({}), but is {} instead.", vocabulary_size, idx);

    return native_->IdToPiece(conditional_cast<int>(idx));
}

std::string
sp_processor::serialize() const
{
    return native_->model_proto().SerializeAsString();
}

}  // namespace fairseq2n::detail
