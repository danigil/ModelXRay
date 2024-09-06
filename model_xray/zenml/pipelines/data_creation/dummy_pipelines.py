from typing import Type
import numpy as np
from zenml import ExternalArtifact, pipeline, step

from model_xray.zenml.pipelines.pipeline_utils import ret_pipeline_with_zenml_model_pp_image_lineage
from model_xray.zenml.steps.dl_model_utils import extract_dl_model_weights_step
from model_xray.configs.models import *

from model_xray.zenml.steps.fetch_pretrained_model import fetch_pretrained_model_step

from model_xray.options import model_collections

@step(enable_cache=True)
def generate_dummy_array_step(data_shape=(100, 100)):
    w=np.random.rand(*data_shape)

    return w

@step(enable_cache=True)
def _extract_weights_dummy_step(preprocessed_image_lineage_config: PreprocessedImageLineage, w_shape=(100, 100)):
# def _extract_weights_dummy_step(pretrained_model_config: CoverDataConfig, w_shape=(100, 100)):
    w = generate_dummy_array_step(data_shape=w_shape)

    return w

# from zenml.materializers.base_materializer import BaseMaterializer
# from zenml.materializers.pydantic_materializer import PydanticMaterializer
# from zenml.enums import ArtifactType

# class MyMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (PreprocessedImageLineage,)
#     ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

#     def load(self, data_type: Type[PreprocessedImageLineage]) -> PreprocessedImageLineage:
#         return PydanticMaterializer(self.uri).load(data_type=PreprocessedImageLineage)

#     def save(self, my_obj: PreprocessedImageLineage) -> None:
#         return PydanticMaterializer(self.uri).save(my_obj)

# from zenml.materializers.materializer_registry import materializer_registry
# materializer_registry.register_and_overwrite_type(key=PreprocessedImageLineage, type_=MyMaterializer)


@pipeline(enable_cache=True)
def _extract_weights_dummy_pipeline(preprocessed_image_lineage_config: PreprocessedImageLineage, w_shape=(100, 100)):
    w = _extract_weights_dummy_step(preprocessed_image_lineage_config=preprocessed_image_lineage_config.model_dump(), w_shape=w_shape)

    return w

extract_weights_dummy_pipeline = ret_pipeline_with_zenml_model_pp_image_lineage(
    pipeline=_extract_weights_dummy_pipeline,
    enable_cache=True
)
# extract_weights_dummy_pipeline = _extract_weights_dummy_pipeline

if __name__ == "__main__":
    # model_names = model_collections['famous_le_100m'].union(model_collections['famous_le_10m'])
    # model_names = ['MobileNet', 'VGG16']
    # xs = range(1,24)
    # xs = [1, 2]
    # model_names = ['ddd', 'cccc', 'aaa']
    # xs = [1, 2]
    # model_names = ['aaron', 'david']
    # xs = [999, 10]
    model_names = ['model1',]
    xs = [1,]

    for i, model_name in enumerate(model_names):
        pm_cfg = PretrainedModelConfig(name=model_name, repo=ModelRepos.KERAS,)

        print(f'!! starting dummy run {i+1}/{len(model_names)} for model {model_name}')

        for ix, x in enumerate(xs):
            print(f'\t@@ starting dummy run {ix+1}/{len(xs)}')

            x_lsb_attack_cfg = XLSBAttackConfig(x=x, fill=True, msb=False)

            preprocessed_image_lineage_config = PreprocessedImageLineage(
                cover_data_config= CoverDataConfig(cover_data_cfg=pm_cfg),
                image_rep_config = ImageRepConfig(),
                image_preprocess_config= ImagePreprocessConfig(image_height=224, image_width=224),
                embed_payload_config= EmbedPayloadConfig(embed_payload_type=PayloadType.RANDOM, embed_proc_config=x_lsb_attack_cfg),
            )

            # _extract_weights_dummy_step.
            extract_weights_dummy_pipeline(preprocessed_image_lineage_config=preprocessed_image_lineage_config)

            print(f'\t@@~~ finished dummy run {ix+1}/{len(xs)}')

        print(f'!!~~ finished dummy run {i+1}/{len(model_names)} for model {model_name}')


