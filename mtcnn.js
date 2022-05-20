require('@tensorflow/tfjs-node')

const tf = require('@tensorflow/tfjs')
const { validateUpdateShape } = require('@tensorflow/tfjs-core/dist/ops/scatter_nd_util')

require('tfjs-npy')

class StageStatus{

    constructor (pad_result=NaN, width=0, height=0){
        this.pad_result = pad_result;
        this.width = width;
        this.height = height;

        if (pad_result != NaN){
            this.update(pad_result)
        }
    }

    update(pad_result){
        this.dy, this.edy, this.dx, this.edx, this.y, 
        this.ey, this.x, this.ex, this.tmpw, this.tmph = pad_result
    }

}

class MTCNN{
    constructor(weight_file = NaN, min_face_size = 20, steps_threshold = NaN, scale_factor = 0.709){

        if (steps_threshold == NaN){
            steps_threshold = [0.6, 0.7, 0.7]
        }

        if (weight_file == NaN){
            weight_file = './mtcnn_weights.npy'
        }

        this.min_face_size = min_face_size
        this.steps_threshold = steps_threshold
        this.scale_factor = scale_factor

        this.pnet, this.rnet, this.onet = this.load_mtcnn_model(weight_file)

    }
}

const = MTCNN()