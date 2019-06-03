package com.shailesh.ml.simplemnistandroid;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
public class MainActivity extends AppCompatActivity {


    ImageView imageView;
    TextView textView;


    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/optimized_frozen_mnist_model.pb";
    private static final String INPUT_NODE = "x_input"; //These from tensorflow model (Python)
    private static final int[] INPUT_SHAPE = {1,784};
    private static final String OUTPUT_NODE = "y_actual"; //based on the tensorflow model and its node wriiten uisng python

    private TensorFlowInferenceInterface inferenceInterface;


    private int imageListIndex = 16;
    private final int[] imageIDList = {
            R.drawable.img_1,
            R.drawable.img_2,
            R.drawable.img_3,
            R.drawable.img_4,
            R.drawable.img_5,
            R.drawable.img_6,
            R.drawable.img_7,
            R.drawable.img_8,
            R.drawable.img_9,
            R.drawable.img_10,
            R.drawable.img_11,
            R.drawable.img_12,
            R.drawable.img_13,
            R.drawable.img_14,
            R.drawable.img_15,
            R.drawable.img_16
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Set up the UI elements
        imageView = (ImageView) findViewById(R.id.image_view);
        textView = (TextView) findViewById(R.id.text_view);

        // Initialize the inference variable to use our model
        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);
    }

    public void predictDigitClick(View view){
        float[] pixelBuffer = convertImage();
        float[] results = formPrediction(pixelBuffer);

        for (float result: results){
            Log.d("dskfj", String.valueOf(result));
        }
        printResults(results);

    }

    private void printResults(float[] results) {
        float max = 0;
        float secondMax = 0;
        int maxIndex = 0;
        int secondMaxIndex = 0;
        for(int i = 0; i < 10; i++) {
            if (results[i] > max) {
                secondMax = max;
                secondMaxIndex = maxIndex;
                max = results[i];
                maxIndex = i;
            } else if (results[i] < max && results[i] > secondMax) {
                secondMax = results[i];
                secondMaxIndex = i;
            }
        }
        String output = "Model predicts: " + String.valueOf(maxIndex) +
                ", second choice: " + String.valueOf(secondMaxIndex);
        textView.setText(output);
    }


    private float[] formPrediction(float[] pixelBuffer) {
        // Fill the input node with the pixel buffer
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SHAPE, pixelBuffer);
        // Make the prediction by running inference on our model and store results in output node
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});
        float[] results = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        // Store value of output node (results) into a float array
        inferenceInterface.readNodeFloat(OUTPUT_NODE, results);
        return results;
    }

    private float[] convertImage(){
        Bitmap imageBitmap = BitmapFactory.decodeResource(getResources(), imageIDList[imageListIndex]);
        imageBitmap = Bitmap.createScaledBitmap(imageBitmap, 28, 28, true);
        imageView.setImageBitmap(imageBitmap);

        int[] imageAsIntArray = new int[784];
        float[] imageAsFloatArray = new float[784];

        imageBitmap.getPixels(imageAsIntArray, 0, 28, 0, 0, 28, 28);

        for (int i=0; i<784; i++){
            imageAsFloatArray[i] = imageAsIntArray[i]/-16777216;
        }
        return imageAsFloatArray;
    }

    public  void loadNextImageClick(View view){
        if(imageListIndex >= 16){
            imageListIndex = 0;
        }
        else{
            imageListIndex += 1;
        }

        imageView.setImageDrawable(getDrawable(imageIDList[imageListIndex]));
    }
}
