# Create the docker image

Start from the respective deployment folder, e.g., `dc_available_qa` or `dc_available_agents`.

Clean artifacts from previous builds. You can skip this step if you are building the docker image for the first time or you are only building this docker image.
```sh
task pre-build-clean
```

Download the required models and make them available to docker
```sh
task pre-build-setup
```

Build the docker image
```sh
task build
```

### Deploy the docker locally

Run the docker with azure enabled on your computer
```sh
task run
```

The demo is now available at http://localhost:8501

### Deploy the docker image to Azure

```sh
task tag
task push
```

The demo is now available at pre-configured url. For now, these are:
- https:.... for dc_available_agents

The url and more can be configured in our AIM-205 azure subscription.
