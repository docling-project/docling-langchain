"""Register Docling plugins."""


def picture_description():
    """Picture description plugins."""
    from langchain_docling.picture_description import PictureDescriptionLangChainModel

    return {
        "picture_description": [
            PictureDescriptionLangChainModel,
        ]
    }
