package is.ispit;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.exam.NeurophExam;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/*
    ARI
 */
public class MojZadatak implements NeurophExam, LearningEventListener {

    int inputCount = 30;
    int outputCount = 1;
    DataSet trainSet;
    DataSet testSet;
    double[] learRate = {0.2, 0.4, 0.6};
    double learningRate;
    int[] hiddenNeurons = {10, 20, 30};
    int hiddenNeuron;
    int numOfIterations = 0;
    int numOfTrainings = 0;
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

        for (double lr : learRate) {
            learningRate = lr;
            for (int hn : hiddenNeurons) {
                hiddenNeuron = hn;
                MultiLayerPerceptron neuralNet = createNeuralNetwork();
                trainNeuralNetwork(neuralNet, ds);
            }
        }

        System.out.println("Srednja vrednost broja iteracija je: " + (double) numOfIterations / numOfTrainings);

        saveBestNetwork();
    }

    @Override
    public DataSet loadDataSet() {
        DataSet dataSet = DataSet.createFromFile("breast_cancer_data.csv", inputCount, outputCount, ",");
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
        return ds.split(0.65, 0.35);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(inputCount, hiddenNeuron, outputCount);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {

        MomentumBackpropagation learningRule = (MomentumBackpropagation) mlp.getLearningRule();
        learningRule.addListener(this);

        learningRule.setLearningRate(learningRate);
        learningRule.setMomentum(0.7);
        learningRule.setMaxIterations(1000);

        mlp.learn(trainSet);

        numOfTrainings++;
        numOfIterations += learningRule.getCurrentIteration();

        evaluate(mlp, testSet);

        return mlp;

    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {

        //pravimo ukupnu gresku i srednju gresku
        double sumError = 0, mse;

        //za svaki red u test setu
        for (DataSetRow row : ds) {
            // setujemo input za neuronsku mrezu
            // mreza uzima karakteristike (inputCount)
            // i na osnovu toga racuna koji je output
            // posle proveravamo da li tacno predvidela vrednost
            mlp.setInput(row.getInput());
            mlp.calculate();

            // uzimamo vrednost za stvarne i predvidjene vrednosti
            double[] actual = row.getDesiredOutput();
            double[] predicted = mlp.getOutput();

            // sumiramo sve vrednosti greske, ovako se racuna greska
            sumError += (double) Math.pow((actual[0] - predicted[0]), 2);

            /*
                za vise outputa prolazimo kroz sve outpute
                i dodajemo gresku u sumu

                for (int i = 0; i < outputCount; i++) {
                    sumMse += (double) Math.pow((actual[i] - predicted[i]), 2);
                }
             */
        }
        //ovako se racuna totalError u Evaluation -> MeanSquaredError klasi
        mse = (double) sumError / (2 * testSet.size());
        System.out.println("\nSrednja kvadratna greska: " + mse + "\n");

        Training t = new Training(mlp, mse);
        trainings.add(t);
    }

    @Override
    public void saveBestNetwork() {
        Training minTraining = trainings.get(0);
        for (Training training : trainings) {
            if (training.getMse() < minTraining.getMse()) {
                minTraining = training;
            }
        }
        minTraining.getNeuralNet().save("nn.nnet");
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
