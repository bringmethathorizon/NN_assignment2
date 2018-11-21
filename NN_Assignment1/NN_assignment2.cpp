#include <iostream>
#include "randlib.h"
#include "mnist/mnist.h"
#include <cmath>

#define numOfInputNodes 785
#define numOfOutputNodes 785
#define numOfEpochs 5
#define numOfHiddenLayers 11
#define epochs 10


void outputCalculation(float outputLayer[], float inputLayer[], float weightsContainer[], int inputLayerSize,
                       int outputLayerSize);

void outputFromInput(float outputLayer[], int inputLayer[], float weightsContainer[], int inputLayerSize,
                     int outputLayerSize);

void squashingFunction(float resultOutput[], int arrayLength);

void calculateErrorForOutput(float errorContainer[], float target[], float outputLayer[], int outputLayerSize);

void calculateErrorForHidden(float hiddenLayerErrorContainer[], float outputErrorContainer[], float weightForHiddenLayers[],
                        float hiddenLayer[], int hiddenLayerSize, int errorContSize);

void createTarget(float target[], int container[], int numberOnPicture);

void weightsUpdate(float weightContainer[], int rows, int cols, float layer[], float error[], float learningRate);

void convertIntToFloat(float float_array[], int array[], int array_length);

void drawTestOutput(float outputArr[]);

using namespace std;

int inputArr[numOfInputNodes];

int main(int argc, char *argv[]) {

    seed_randoms();
    float sampNoise = rand_frac() / 2.0; // sets default sampNoise

    // --- a simple example of how to set params from the command line
    if (argc == 2) { // if an argument is provided, it is SampleNoise
        sampNoise = atof(argv[1]);
        if (sampNoise < 0 || sampNoise > .5) {
            printf("Error: sample noise should be between 0.0 and 0.5\n");
            return 0;
        }
    }
    // --- an example for how to work with the included mnist library:
    mnist_data *zData;      // each image is 28x28 pixels
    unsigned int sizeData; // depends on loadType
    unsigned int sizeData2; // depends on loadType

    int loadType = 2; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData, &sizeData, loadType)) {
        printf("something went wrong loading data set\n");
        return -1;
    }

    mnist_data *zData2;      // each image is 28x28 pixels
    unsigned int sizeData1;  // depends on loadType
    int loadType1 = 2; // loadType may be: 0, 1, or 2
    if (mnistLoad(&zData2, &sizeData2, loadType1)) {
        printf("something went wrong loading data set\n");
        return -1;
    }

    float learningRate = 0.1;

    //    int pictureLabel = 6;

    //    int inputArr[numOfInputNodes];

    float outputArr[numOfOutputNodes];

    float target[numOfOutputNodes];

    float inputLayerArr[numOfInputNodes];

    float hiddenLayerArr[numOfHiddenLayers];

    float outputErrors[numOfOutputNodes];

    float hiddenErrors[numOfHiddenLayers];



    //counting how many weights we have from input layer to hidden and from hidden layer to output layer

    //    int amountOfWeightsInputLr = numOfInputNodes*numOfHiddenLayers;
    //    int amountOfWeightsHiidenLr = numOfHiddenLayers*numOfOutputNodes;


    //initializing weights from hidden to input with random values
    float weightsInputHidden[numOfInputNodes * numOfHiddenLayers];

    for (int i = 0; i < numOfHiddenLayers; i++) {
        for (int ii = 0; ii < numOfInputNodes; ii++) {
            weightsInputHidden[(i * numOfInputNodes) + ii] = rand_weight();
        }
    }

    //initializing weights from hidden to output with random values
    float weightsHiddenOutput[numOfHiddenLayers * numOfOutputNodes];
    for (int i = 0; i < numOfHiddenLayers; i++) {
        for (int ii = 0; ii < numOfOutputNodes; ii++) {
            weightsHiddenOutput[(i * numOfOutputNodes) + ii] = rand_weight();
        }
    }



    //training

    for (int ep = 0; ep < 5; ep++) {

        //testing
//        cout << "TESTING" << endl;
//
//        for (int picIndex = 0; picIndex < sizeData; picIndex++) {
//
//            get_input(inputArr, zData, picIndex, sampNoise);
//
//            createTarget(target, inputArr, zData[picIndex].label);
//
//            outputFromInput(hiddenLayerArr, inputArr, weightsInputHidden, numOfInputNodes, numOfHiddenLayers);
//            squashingFunction(hiddenLayerArr, numOfHiddenLayers);
//
//            hiddenLayerArr[0] = 1;
//
//            outputCalculation(outputArr, hiddenLayerArr, weightsHiddenOutput, numOfHiddenLayers, numOfOutputNodes);
//            squashingFunction(outputArr, numOfOutputNodes);
//
//
//        }

        //learning

        for (int pic = 0; pic < sizeData2; pic++) {

            get_input(inputArr, zData2, pic, sampNoise);

            createTarget(target, inputArr, zData[pic].label);

            outputFromInput(hiddenLayerArr, inputArr, weightsInputHidden, numOfInputNodes, numOfHiddenLayers);
            squashingFunction(hiddenLayerArr, numOfHiddenLayers);

            hiddenLayerArr[0] = 1;

            outputCalculation(outputArr, hiddenLayerArr, weightsHiddenOutput, numOfHiddenLayers, numOfOutputNodes);
            squashingFunction(outputArr, numOfOutputNodes);

//            drawTestOutput(outputArr);

            calculateErrorForOutput(outputErrors, target, outputArr, numOfOutputNodes);

            calculateErrorForHidden(hiddenErrors, outputErrors, weightsHiddenOutput, hiddenLayerArr, numOfHiddenLayers,
                                    numOfOutputNodes);

            weightsUpdate(weightsHiddenOutput, numOfOutputNodes, numOfHiddenLayers, hiddenLayerArr, outputErrors,
                          learningRate);

            weightsUpdate(weightsInputHidden, numOfHiddenLayers, numOfInputNodes, inputLayerArr, hiddenErrors,
                          learningRate);
        }

    }

    return 0;
}


//create target
void createTarget(float target[], int container[], int numberOnPicture) {
    for (int i = 0; i < numOfOutputNodes; i++) {
        target[i] = (float) container[i];
    }
}

//multiplying weights with nodes(hidden --> input)

void outputFromInput(float outputLayer[], int inputLayer[], float weightsContainer[], int inputLayerSize,
                     int outputLayerSize) {
    for (int i = 0; i < outputLayerSize; i++) {
        float counter = 0;
        for (int ii = 0; ii < inputLayerSize; ii++) {
            counter += weightsContainer[(i * inputLayerSize) + ii] * inputLayer[ii];
        }
        outputLayer[i] = counter;
    }
}

//multiplying weights with nodes(hidden --> output)

void outputCalculation(float outputLayer[], float inputLayer[], float weightsContainer[], int inputLayerSize,
                       int outputLayerSize) {
    for (int i = 0; i < outputLayerSize; i++) {
        float counter = 0;
        for (int ii = 0; ii < inputLayerSize; ii++) {
            counter += weightsContainer[(i * inputLayerSize) + ii] * inputLayer[ii];
        }
        outputLayer[i] = counter;
//        cout << "output[" << i << "] = " << outputLayer[i] << endl;
    }
}

//squash multiplication results in order to get values from range [0;1]

void squashingFunction(float resultOutput[], int arrayLength) {
    for (int i = 0; i < arrayLength; i++) {
        resultOutput[i] = 1.0 / (1.0 + exp(-1 * resultOutput[i]));
//        cout << "squashed_output[" << i << "] = " << resultOutput[i] << endl;
    }
}

//calculate errors in output layer

void calculateErrorForOutput(float errorContainer[], float target[], float outputLayer[], int outputLayerSize) {
    for (int i = 0; i < outputLayerSize; i++) {
        errorContainer[i] = (target[i] - outputLayer[i]) * outputLayer[i] * (1 - outputLayer[i]);
//        cout << "errorContainer[" << i << "] = " << errorContainer[i] << endl;
    }
}

//calculate errors in hidden layer

void
calculateErrorForHidden(float hiddenLayerErrorContainer[], float outputErrorContainer[], float weightForHiddenLayers[],
                        float hiddenLayer[], int hiddenLayerSize, int errorContSize) {
    for (int i = 0; i < hiddenLayerSize; i++) {
        float multSum = 0;
        for (int ii = 0; ii < errorContSize; ii++) {
            multSum += outputErrorContainer[ii] * weightForHiddenLayers[ii * hiddenLayerSize + i];
        }

        hiddenLayerErrorContainer[i] = hiddenLayer[i] * (1 - hiddenLayer[i]) * multSum;
    }
}

//updating weights

void weightsUpdate(float weightContainer[], int rows, int cols, float layer[], float error[], float learningRate) {

    float deltaWeightArray[rows * cols];

    for (int i = 0; i < 785; i++) {
        for (int j = 0; j < 10; j++) {

            deltaWeightArray[i * cols + j] = error[i] * layer[j] * learningRate;

            weightContainer[i * cols + j] += deltaWeightArray[i * cols + j];

            cout << "weightContainer[" << i << "][" << j << "] = "  << weightContainer[i] << endl;
        }
    }
}

void convertIntToFloat(float float_array[], int array[], int array_length) {

    for (int i = 0; i < array_length; i++) {
        float_array[i] = (float) array[i];
    }
}

void drawTestOutput(float outputArr[]) {

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {

            if ((outputArr[i * 28] + j + 1) >= 0.5) {
                cout << "X";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }

    cout << endl;
}