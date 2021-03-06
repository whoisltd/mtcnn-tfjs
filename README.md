# mtcnn-tfjs
Implement mtcnn with Tensorflow.js

[![CodeFactor](https://www.codefactor.io/repository/github/whoisltd/mtcnn-tfjs/badge)](https://www.codefactor.io/repository/github/whoisltd/mtcnn-tfjs)
## What is this?

A face detection framework with MTCNN and Tensorflow.js

Give me a ⭐️, if you like it ❤️

(Currently, framework is only accepted to detect one face, i'll update soon)

## Installation

Run:
```
npm install whoisltd/mtcnn-tfjs
```

Use:
 ```node
const mtcnn = require('@whoisltd/mtcnn-tfjs');

const pnet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/pnet/model.json'

const rnet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/rnet/model.json'

const onet_url = 'https://storage.googleapis.com/my-mtcnn-models/final_model/onet/model.json'

let model = null
if(model == null){
    model = new mtcnn.MTCNN(pnet_url, rnet_url, onet_url)
    model.mtcnn.pnet = await model.mtcnn.pnet
    model.mtcnn.rnet = await model.mtcnn.rnet
    model.mtcnn.onet = await model.mtcnn.onet
}

//Draw bounding boxes on image:
model.draw_img(url_img, url_output);

//Crop face from image:
model.crop_face(url_img, url_output, true);

//Get bounding boxes, landmarks, score:
model.detect(url_img);

```

## Demo

<p align="center"><img src="https://raw.githubusercontent.com/whoisltd/mtcnn-tfjs/master/images/result.png" width="50%" height="50%"></p>

## Contribution
Pull request is welcome!
