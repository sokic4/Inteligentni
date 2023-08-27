/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package is.ispit;

import org.neuroph.core.NeuralNetwork;

/**
 *
 * @author Ari
 */
public class Training {
    
    private NeuralNetwork neuralNet;
    private double accuracy;

    public Training(NeuralNetwork neuralNet, double accuracy) {
        this.neuralNet = neuralNet;
        this.accuracy = accuracy;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public NeuralNetwork getNeuralNet() {
        return neuralNet;
    }

    public void setNeuralNet(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }
    
    
    
}
