from __future__ import annotations

from typing import Final
from dataclasses import dataclass, field


from fairseq2.models.bestrq.masker import RandomNoiseMasker
from fairseq2.models.bestrq.model import BestRQModel
from fairseq2.models.bestrq.quantizer import MultiRandomVectorQuantizer
from fairseq2.models.wav2vec2.factory import Wav2Vec2EncoderBuilder
from fairseq2.config_registry import ConfigRegistry


from fairseq2.models.feature_extractor import SequenceFeatureExtractor
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderBuilder, Wav2Vec2FeatureExtractor, Wav2Vec2FbankFeatureExtractor
from fairseq2.models.wav2vec2.vector_quantizer import VectorQuantizer
from fairseq2.typing import DataType, Device
from fairseq2.models.factory import model_factories

BESTRQ_FAMILY: Final = "bestrq"

from dataclasses import field
from math import prod

from fairseq2.models.wav2vec2.factory import Wav2Vec2EncoderConfig

@dataclass(kw_only=True)
class BestRQConfig:
    """Holds the configuration of a BestRQ model.

    The default values correspond to the base architecture as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.
    """

    model_type: str = "bestrq"

    encoder_config: Wav2Vec2EncoderConfig = field(
        default_factory=lambda: Wav2Vec2EncoderConfig()
    )
    """The configuration of the bestrq encoder, which is same as the config for the wav2vec2 model"""
    
    final_dim: int = 256
    """The dimensionality of the final projection that is applied to context
    network outputs and quantized targets."""

    label_smoothing: float = 0.0

    # Mask
    temporal_mask_span_len: int = 10
    """The length of each temporal mask span that is applied over time steps."""

    max_temporal_mask_prob: float = 0.69
    """The maximum probability of masking a time step. Note that, due to mask
    span overlap, the effective probability will be lower."""

    min_num_temporal_mask_spans: int = 2
    """The minimum number of temporal masks sampled per sequence."""

    spatial_mask_span_len: int = 10
    """The length of each spatial mask span that is applied over features."""

    max_spatial_mask_prob: float = 0.0
    """The maximum probability of masking a feature. Note that, due to mask span
    overlap, the effective probability will be lower."""

    min_num_spatial_mask_spans: int = 2
    """The minimum number of spatial masks sampled per sequence."""

    noise_mean: float = 0.0
    """Mean of the normal distribution used to sample the noise for the masked elements"""

    noise_standard_deviation: float = 0.1
    """Standard deviation of the normal distribution used to sample the noise for the masked elements"""

    # Quantization
    num_quantizer: int = 2
    """Number of quantizer used in training"""

    quantized_dim: int = 256
    """The output dimensionality of vector quantizer."""

    num_codebook_entries: int = 320
    """The number of entries per codebook."""

    normalize_quantizer_input: bool = field(
        default=False,
        metadata={
            "help": "Whether to normalize the input to the quantizer",
        },
    )


class BestRQBuilder:
    """Builds modules of a wav2vec 2.0 model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2006.11477`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    _config: BestRQConfig
    _encoder_builder: Wav2Vec2EncoderBuilder
    _device: Device | None
    _dtype: DataType | None

    def __init__(
        self,
        config: BestRQConfig,
        encoder_builder: Wav2Vec2EncoderBuilder | None = None,
        *,
        device: Device | None = None,
        dtype: DataType | None = None,
    ) -> None:
        """
        :param config:
            The configuration.
        :param encoder_builder:
            The encoder builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self._config = config
        
        if encoder_builder is None:
            encoder_builder = Wav2Vec2EncoderBuilder(
                config.encoder_config, device=device, dtype=dtype
            )
            
        self._encoder_builder = encoder_builder

        self._device, self._dtype = device, dtype

    def build_model(self) -> BestRQModel:
        """Build a model."""
        encoder_frontend = self._encoder_builder.build_frontend()

        encoder = self._encoder_builder.build_encoder()

        masker = self.build_masker()

        quantizer = self.build_quantizer()

        downsampler = self.build_downsampler()

        return BestRQModel(
            encoder_frontend,
            encoder,
            masker,
            quantizer,
            downsampler,
            self._config.final_dim,
            device=self._device,
            dtype=self._dtype,
        )

    def build_masker(self) -> RandomNoiseMasker:
        """Build a feature masker."""
        return RandomNoiseMasker(
            self._config.encoder_config.model_dim,
            self._config.temporal_mask_span_len,
            self._config.max_temporal_mask_prob,
            self._config.min_num_temporal_mask_spans,
            self._config.spatial_mask_span_len,
            self._config.max_spatial_mask_prob,
            self._config.min_num_spatial_mask_spans,
            self._config.noise_mean,
            self._config.noise_standard_deviation,
            device=self._device,
            dtype=self._dtype,
        )

    def build_quantizer(self) -> VectorQuantizer:
        """Build a vector quantizer."""
        return MultiRandomVectorQuantizer(
            self._config.encoder_config.feature_dim,
            self._config.quantized_dim,
            self._config.num_codebook_entries,
            self._config.num_quantizer,
            self._config.normalize_quantizer_input,
            device=self._device,
            dtype=self._dtype,
        )

    def build_downsampler(self):
        
        if self._config.encoder_config.use_fbank:
            conv_downsampler = Wav2Vec2FbankFeatureExtractor(
                self._config.encoder_config.num_fbank_channels,
                self._config.encoder_config.fbank_stride,
                sample_every_k=self._config.encoder_config.sample_fbank_every_k,
            )
        else:
            conv_downsampler = Wav2Vec2FeatureExtractor(
                self._config.encoder_config.feature_extractor_layer_descs,
                self._config.encoder_config.feature_extractor_bias,
                layer_norm=self._config.encoder_config.feature_extractor_layer_norm_convs,
                gradient_scale=self._config.encoder_config.feature_gradient_scale,
                device=self._device,
                dtype=self._dtype,
            )
        for param in conv_downsampler.parameters():
            param.requires_grad = False
        return conv_downsampler

def create_bestrq_model(
    config: BestRQConfig,
    *,
    device: Device | None = None,
    dtype: DataType | None = None,
) -> BestRQModel:
    """Create a wav2vec 2.0 model."""
    return BestRQBuilder(config, device=device, dtype=dtype).build_model()


bestrq_encoder_archs = ConfigRegistry[Wav2Vec2EncoderConfig]()
bestrq_encoder_arch = bestrq_encoder_archs.decorator

bestrq_archs = ConfigRegistry[BestRQConfig]()
bestrq_arch = bestrq_archs.decorator


model_factories.register(
    BESTRQ_FAMILY, create_bestrq_model, BestRQConfig, bestrq_archs
)
