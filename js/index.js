


class Newron {
    static heavyside = (input) => input >= 0 ? 1 : 0;

    constructor(weightsSize, activation, learningRate) {
        this.weights = Array.from({ length: weightsSize }, () => Math.random());
        this.activation = activation;
        this.learningRate = learningRate;
        this.bias = 0;
    }

    fit = (xTrain, yTrain) => {
    
        for (let i = 0; i < yTrain.length; i++) {
            const x = xTrain[i];
            const y_hat = this.predict(x);

            if (yTrain[i] === y_hat) continue;

            const error = this.error(yTrain[i], y_hat);

            console.log('Error: ', error);

            this.updateWeigths(x, error);
            this.updateBias(error)
        }
    }

    updateWeigths = (x, error) => {
        const { weights, learningRate } = this; 

        for (let i = 0; i < weights.length; i++) {
            weights[i] = weights[i] + learningRate * x[i] * error;
        } 
    }

    updateBias = (error) => {
        this.bias = this.bias + this.learningRate * error;
    }

    predict = (input) => {
        if (input.length > this.weights.length) throw new Error("Wrong size for input");

        let wSum = this.bias;

        for (let i = 0; i < this.weights.length; i++) {
            wSum += this.weights[i] * input[i];
        }

        return this.activation(wSum);
    }

    error = (y, y_hat) => {
        return y - y_hat;        
    }
}

// class Perceptron {
//     constructor(neurons) {
//         this.neurons = neurons;
//     }

//     updateWeigths = (x, error) => {
//         for (const neuron of this.neurons) {
//             const { weights, learningRate } = neurons; 

//             for (let i = 0; i < weights.length; i++) {
//                 weights[i] = weights[i] + learningRate * x[i] * error;
//             } 
//         }
//     }
// }

module.exports = Newron
