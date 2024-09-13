import itertools


from model_xray.zenml.zenml_lookup import try_get_artifact_preprocessed_image
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *

from model_xray.zenml.pipelines.data_creation.preprocessed_image import preprocessed_image_pipeline

if __name__ == "__main__":
    im_types = [
        ImageRepConfig(
            image_rep_proc_config=GrayscaleFourpartConfig()
        ),
    ]

    im_preprocesses = [
        ImagePreprocessConfig(
            image_height=100,
            image_width=100,
        ),
    ]

    for model_name in maleficnet_cover_model_names:
        print(f'!! starting model_name: {model_name}')
        curr_mal_options = maleficnet_mal_options_map[model_name]

        for mal_name, im_type, im_preprocess \
        in itertools.product(
            curr_mal_options + ['NA',],
            im_types,
            im_preprocesses,
        ):
            curr_embed_payload_config = ret_na_val()
            if mal_name != 'NA':
                payload_filepath = get_maleficnet_payload_filepath(mal_name)

                curr_embed_payload_config = EmbedPayloadConfig(
                    embed_payload_type=PayloadType.BINARY_FILE,
                    embed_proc_config=MaleficnetAttackConfig(
                        malware_path_str=payload_filepath
                    ),
                    embed_payload_metadata=EmbedPayloadMetadata(
                        payload_filepath=payload_filepath
                    )
                )

            pp_img_lineage = PreprocessedImageLineage(
                cover_data_config=CoverDataConfig(
                    cover_data_cfg=MaleficnetCoverModelConfig(
                        name=model_name,
                    )
                ),
                image_rep_config=im_type,
                image_preprocess_config=im_preprocess,
                embed_payload_config=curr_embed_payload_config
            )

            print(f'\t@@ starting \n{pp_img_lineage.model_dump(mode="json")}\n')

            try:
                artifact_lookup = try_get_artifact_preprocessed_image(pp_img_lineage)
                print(f'\t## found artifact, skipping pipeline execution')
            except Exception as e:
                print(f'\t%% didn\'t find artifact')
                preprocessed_image_pipeline(pp_img_lineage)