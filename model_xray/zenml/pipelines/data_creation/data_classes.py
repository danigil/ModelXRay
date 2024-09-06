import os
from typing import Dict, Type

import h5py
import numpy as np
import numpy.typing as npt


from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import VisualizationType
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.materializers.numpy_materializer import NumpyMaterializer

class MZWeights:
    model_zoo_name: str
    data: np.ndarray

NUMPY_Z_FILENAME = "data.npz"

class MZWeightsMaterializer(BaseMaterializer):

    ASSOCIATED_TYPES = (MZWeights,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self) -> Type[MZWeights]:
        """Reads a numpy array from a `.npy` file.

        Args:
            data_type: The type of the data to read.


        Raises:
            ImportError: If pyarrow is not installed.

        Returns:
            The numpy array.
        """

        numpy_file = os.path.join(self.uri, NUMPY_Z_FILENAME)

        if self.artifact_store.exists(numpy_file):
            with self.artifact_store.open(numpy_file, "rb") as f:
                arr = np.load(f, allow_pickle=True)

            return MZWeights(data=arr['data'])
        else:
            raise FileNotFoundError(f"File {numpy_file} not found / artificat store not found")

    def save(self, mzw: Type[MZWeights]) -> None:
        """Writes a MZWeights to the artifact store as a `.npz` file.

        Args:
            mzw: The MZWeights to write.
        """
        with self.artifact_store.open(
            os.path.join(self.uri, NUMPY_Z_FILENAME), "wb"
        ) as f:
            np.savez_compressed(f, kwds = {'data': mzw.arr})

    # def save_visualizations(
    #     self, mzw: Type[MZWeights]
    # ) -> Dict[str, VisualizationType]:
    #     return super().save_visualizations(mzw.data)


