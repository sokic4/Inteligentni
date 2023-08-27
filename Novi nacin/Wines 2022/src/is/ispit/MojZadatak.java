package is.ispit;

import java.util.ArrayList;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.exam.NeurophExam;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/*
    ARI
 */
public class MojZadatak implements NeurophExam, LearningEventListener {

    int inputCount = 13;
    int outputCount = 3;
    DataSet trainSet;
    DataSet testSet;
    double[] learRate = {0.2, 0.4, 0.6};
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
        return DataSet.createFromFile("wines.csv", inputCount, outputCount, ",");
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
        return ds.split(0.7, 0.3);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(TransferFunctionType.TANH, inputCount, 22, outputCount);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {

        int numOfIterations = 0;
        int numOfTrainings = 0;

        for (double lr : learRate) {
            // getujemo LearningRule iz nase neuronske mreze - algoritam koji nam je receno da koristimo
            // BackPropagation ili MomentumBackPropagation i podesavamo mu momentum/gresku po zadatku
            BackPropagation learningRule = mlp.getLearningRule();
            learningRule.addListener(this);

            learningRule.setLearningRate(lr);
            learningRule.setMaxError(0.02);
            learningRule.setMaxIterations(1000); // ako se program predugo izvrsava

            // sada ucimo nasu mrezu, uci se nad trainSetom
            mlp.learn(trainSet);

            numOfTrainings++;
            numOfIterations += learningRule.getCurrentIteration();

            // testiramo sta je naucila nad testSetom i dobijamo accuracy ili srednju kvadratnu gresku
            evaluate(mlp, testSet);
        }

        System.out.println("Srednja vrednost broja iteracija je: " + (double) numOfIterations / numOfTrainings);

        return mlp;

    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {

        // pravimo class labels onoliko koliko ima output varijabli
        String[] classLabels = new String[]{"c1", "c2", "c3"};
        ConfusionMatrix cm = new ConfusionMatrix(classLabels);
        double accuracy = 0;

        for (DataSetRow dataSetRow : ds) {
            // prolazimo kroz svaki red u testSetu i racunamo izlaz
            // na osnovu input varijabli
            mlp.setInput(dataSetRow.getInput());
            mlp.calculate();

            // dobijamo niz double vrednosti gde je desiredOutput tacno resenje
            // a mlp.getOutput() ono sto je nasa neuronska mreza izracunala
            // pa vadimo index maksimalne vrednosti (vrste u kojoj je vino rasporedjeno)
            // i povecavamo odredjeni element matrice
            int actual = getMaxIndex(dataSetRow.getDesiredOutput());
            int predicted = getMaxIndex(mlp.getOutput());

            cm.incrementElement(actual, predicted);
        }

        // racunamo accuracy po tipicnoj formuli za accuracy sto smo ucili na R-u
        // suma dijagonale, kroz suma svih
        for (int i = 0; i < outputCount; i++) {
            accuracy += (double) (cm.getTruePositive(i) + cm.getTrueNegative(i)) / cm.getTotal();
        }

        System.out.println(cm.toString());

        accuracy = (double) accuracy / outputCount;

        System.out.println("Moj accuracy: " + accuracy);

        // dodajemo u treninge neuronsku mrezu i njen accuracy
        // pa kad zavrsimo sve treninge, cuvamo mrezu sa najvecom tacnoscu
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
        BackPropagation bp = (BackPropagation) le.getSource();
        System.out.println("Iteration: " + bp.getCurrentIteration()
                + " Total network error: " + bp.getTotalNetworkError());
        // prikazuje nam ukupnu gresku za svaku iteraciju
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
