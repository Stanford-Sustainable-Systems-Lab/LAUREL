# Loading Graphhopper Containers in Apptainer

Loading these containers takes a few manual steps, so I'll include them here.
    1. Pull the container in sandbox mode using `apptainer pull --sandbox docker://israelhikingmap/graphhopper:10.2`
    2. Edit the container's runscript to run `cd /graphhopper` right at the beginning. This gets around some of the incompatibilities between the Graphhopper containers' Dockerfile's use of `WORKDIR`, which Apptainer does not support.
    3. Edit the main entrypoint script at `/graphhopper/graphhopper.sh` to search for the `.jar` file only within the `/graphhopper` directory. This is near the bottom of the file where the environment variable `JAR` is being set.