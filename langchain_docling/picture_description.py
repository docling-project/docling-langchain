"""Picture description model using LangChain primitives."""

import base64
import io
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal, Optional, Type, Union

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PictureDescriptionBaseOptions
from docling.models.picture_description_base_model import PictureDescriptionBaseModel
from docling.models.utils.hf_model_download import HuggingFaceModelDownloadMixin
from langchain_core.language_models.chat_models import BaseChatModel
from PIL import Image


class PictureDescriptionLangChainOptions(PictureDescriptionBaseOptions):
    """Options for the PictureDescriptionLangChainModel."""

    kind: ClassVar[Literal["langchain"]] = "langchain"
    llm: BaseChatModel
    prompt: str = "Describe this document picture in a few sentences."
    provenance: Optional[str] = None


class PictureDescriptionLangChainModel(
    PictureDescriptionBaseModel, HuggingFaceModelDownloadMixin
):
    """Implementation of a PictureDescription model using LangChain."""

    @classmethod
    def get_options_type(cls) -> Type[PictureDescriptionBaseOptions]:
        """Define the option type for the factory."""
        return PictureDescriptionLangChainOptions

    def __init__(
        self,
        enabled: bool,
        enable_remote_services: bool,
        artifacts_path: Optional[Union[Path, str]],
        options: PictureDescriptionLangChainOptions,
        accelerator_options: AcceleratorOptions,
    ):
        """Initialize PictureDescriptionLangChainModel."""
        super().__init__(
            enabled=enabled,
            enable_remote_services=enable_remote_services,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PictureDescriptionLangChainOptions

        if self.enabled:
            self.llm = self.options.llm
            self.provenance = "langchain"
            if self.options.provenance:
                self.provenance += f"-{self.options.provenance}"

    def _annotate_images(self, images: Iterable[Image.Image]) -> Iterable[str]:
        """Annotate the images with the LangChain model."""
        # Create input messages
        batch_messages = []

        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
            batch_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.options.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                },
                            },
                        ],
                    }
                ]
            )

        responses = self.llm.batch(batch_messages)
        for resp in responses:
            yield resp.text
