package is.ispit;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.exam.NeurophExam;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/*
    ARI
*/

public class MojZadatak implements NeurophExam, LearningEventListener {

    int inputCount = 8;
    int outputCount = 1;
    DataSet trainSet;
    DataSet testSet;
    double[] learRate = {0.2, 0.3, 0.4};
    ArrayList<Training> trainings = new ArrayList<>();

    /**
     * U ovoj metodi pozivati sve metode koje cete implementirati iz NeurophExam
     * interfejsa
     */
    private void run() {
        DataSet ds = loadDataSet();
        ds = preprocessDataSet(ds);
        DataSet[] trainAndTest = trainTestSplit(ds);
        trainSet = trainAndTest[0];
        testSet = trainAndTest[1];
        MultiLayerPerceptron neuralNet = createNeuralNetwork();
        trainNeuralNetwork(neuralNet, ds);
        saveBestNetwork();
    }

    @Override
    public DataSet loadDataSet() {
        DataSet dataSet = DataSet.createFromFile("diabetes_data.csv", inputCount, outputCount, ",");
        return dataSet;
    }

    @Override
    public DataSet preprocessDataSet(DataSet ds) {
        Normalizer norm = new MaxNormalizer(ds);
        norm.normalize(ds);
        ds.shuffle();
        return ds;
    }

    @Override
    public DataSet[] trainTestSplit(DataSet ds) {
        return ds.split(0.6, 0.4);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(inputCount, 20, 16, outputCount);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {

        int numOfIterations = 0;
        int numOfTrainings = 0;

        for (double lr : learRate) {
            MomentumBackpropagation learningRule = (MomentumBackpropagation) mlp.getLearningRule();
            learningRule.addListener(this);

            learningRule.setLearningRate(lr);
            learningRule.setMaxError(0.07);
            learningRule.setMomentum(0.5);
//            learningRule.setMaxIterations(1000);

            mlp.learn(trainSet);

            numOfTrainings++;
            numOfIterations += learningRule.getCurrentIteration();

            evaluate(mlp, testSet);
        }

        System.out.println("Srednja vrednost broja iteracija je: " + (double) numOfIterations / numOfTrainings);

        return mlp;

    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {

        // ova matrica ima 2 classLabela jer ima 1 output
        // matrica ne sme da bude 1x1
        String[] classLabels = new String[]{"c1", "c2"};
        ConfusionMatrix cm = new ConfusionMatrix(classLabels);
        double accuracy = 0;

        for (DataSetRow dataSetRow : ds) {
            mlp.setInput(dataSetRow.getInput());
            mlp.calculate();

            int actual = (int) Math.round(dataSetRow.getDesiredOutput()[0]);
            int predicted = (int) Math.round(mlp.getOutput()[0]);
            
            System.out.println("Actual: " + dataSetRow.getDesiredOutput()[0] 
                            + "\t Predicted: " + mlp.getOutput()[0]);

            cm.incrementElement(actual, predicted);
        }

        accuracy = (double) (cm.getTruePositive(0) + cm.getTrueNegative(0)) / cm.getTotal();

        System.out.println(cm.toString());

        System.out.println("Moj accuracy: " + accuracy);

        Training t = new Training(mlp, accuracy);
        trainings.add(t);
    }

    @Override
    public void saveBestNetwork() {
        Training maxTraining = trainings.get(0);
        for (Training training : trainings) {
            if (training.getAccuracy() > maxTraining.getAccuracy()) {
                maxTraining = training;
            }
        }
        maxTraining.getNeuralNet().save("nn.nnet");
    }

    public static void main(String[] args) {
        new MojZadatak().run();
    }

    @Override
    public void handleLearningEvent(LearningEvent le) {
        MomentumBackpropagation bp = (MomentumBackpropagation) le.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration()
                + " Total network error: " + bp.getTotalNetworkError());
    }

    private int getMaxIndex(double[] output) {
        int max = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[max] < output[i]) {
                max = i;
            }
        }
        return max;
    }
 
/*
    ARI
*/

}
