const Perceptron = require('./index')


// const x_or = [
//     [1, 0],
//     [0, 1],
//     [1, 1],
//     [0, 0], 
// ]
// const y = [
//     1,
//     1,
//     1,
//     0
// ]

// const model = new Perceptron(2, Perceptron.heavyside, 0.1);

// model.fit(x_or, y);
// model.fit(x_or, y);
// model.fit(x_or, y);



// console.log(model.predict(x_or[0]))
// console.log(model.predict(x_or[1]))
// console.log(model.predict(x_or[2]))
// console.log(model.predict(x_or[3]))

const x_and = [
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
]

const y_and = [
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0
]

const model = new Perceptron(3, Perceptron.heavyside, 0.1);

model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);
model.fit(x_and, y_and);


console.log(model.predict(x_and[0]))
console.log(model.predict(x_and[1]))
console.log(model.predict(x_and[2]))
console.log(model.predict(x_and[3]))
// console.log(model.weights)
