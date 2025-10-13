# resnet-task-models


### Setup
In order to run, this codebase needs the ResNet model weight files. They are too large to put on GitHub so will be stored separately. Unzip `model-weights.zip` in main directory, so file structure looks as such:

```
.
├── resnet-task-models/
│   ├── model-weights/
│   │   ├── README.md
│   │   └── <date of generated weights>
|   |       ├── task_<x>_final_weights_<date>.pth
│   ├── server.py
│   ├── requirements.txt
│   ├── README.md
│   └── etc...

```

To run: `python3 server.py`.


### `/classify` endpoint (POST)
Sample request


```
Content-Type: application/json
{
    "task_id": Int from 1 to 6 inclusive,
    "image": base64 encoded image
}
```

Output:
```
{
    "completed": boolean
}
```


In the main directory is a script `convert_to_base_image.py`. This can be used to get a base64 representaion of an image, which is then fed into this API call.  
