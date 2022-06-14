# mtcnn-tfjs
Implement mtcnn with Tensorflow.js
## What is this?

A face detection framework with MTCNN and Tensorflow.js

Give me a ⭐️, if you like it ❤️

(Currently, framework is only accepted to detect one face, i'll update soon)

## Installation

Run:
```
npm install @whoisltd/mtcnn-tfjs
```

Use:
 ```node
const mtcnn = require('@whoisltd/mtcnn-tfjs');

//Draw bounding boxes on image:
mtcnn.draw_img(url_img, url_output);

//Crop face from image:
mtcnn.crop_face(url_img, url_output, true);

//Get bounding boxes, landmarks, score:
mtcnn.detect(url_img);

```

## Demo

<p align="center"><img src="https://raw.githubusercontent.com/whoisltd/mtcnn-tfjs/master/images/result.png" width="50%" height="50%"></p>

## Contribution
Pull request is welcome!
