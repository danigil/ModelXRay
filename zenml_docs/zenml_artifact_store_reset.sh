conda activate /home/danielg/micromamba/envs/zenml
cd /home/danielg/danigil/ModelXRay
zenml stack update custom-stack -a default
zenml artifact-store delete custom_local
sudo rm -r /mnt/exdisk2/zenml_artifact_store/*
zenml artifact-store register custom_local --flavor local --path=/mnt/exdisk2/zenml_artifact_store/
zenml stack update custom-stack -a custom_local
zenml stack set custom-stack