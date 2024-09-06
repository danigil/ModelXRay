from typing import Annotated
from zenml import ArtifactConfig, get_step_context, step

import numpy.typing as npt

from model_xray.configs.types import DL_MODEL_TYPE
from model_xray.utils.model_utils import extract_weights as extract_weights_util

@step(enable_cache=True)
def extract_dl_model_weights_step(
    model: DL_MODEL_TYPE,
) -> (
    Annotated[
        np.ndarray,
        ArtifactConfig(
            name="extracted_dl_model_weights",
        ),
    ]
):
  
    w = extract_weights_util(model=model)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="extracted_dl_model_weights",
        metadata={
            "weights_properties": {
                "dtype": str(w.dtype).lower(),
                "amount": len(w)
            }
        }
    )
    step_context.add_output_tags(
        output_name="extracted_dl_model_weights",
        tags=["weights_extracted", "weights_dl_model"]
    )

    print(f"Extracted {len(w)} weights from model")

    return w