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
    private double mse;

    public Training(NeuralNetwork neuralNet, double mse) {
        this.neuralNet = neuralNet;
        this.mse = mse;
    }

    public double getMse() {
        return mse;
    }

    public void setMse(double accuracy) {
        this.mse = accuracy;
    }

    public NeuralNetwork getNeuralNet() {
        return neuralNet;
    }

    public void setNeuralNet(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }
    
    
    
}
