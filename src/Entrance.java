import java.util.Arrays;

import javax.swing.JFrame;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;

import pers.season.vml.statistics.regressor.LearningParams;
import pers.season.vml.statistics.regressor.RegressorSet;
import pers.season.vml.statistics.regressor.RegressorSetInstance;
import pers.season.vml.statistics.regressor.RegressorTrain;
import pers.season.vml.statistics.shape.ShapeInstance;
import pers.season.vml.statistics.shape.ShapeModel;
import pers.season.vml.statistics.shape.ShapeModelTrain;
import pers.season.vml.util.FaceDetector;
import pers.season.vml.util.ImUtils;
import pers.season.vml.util.MuctData;

public class Entrance {
	static {
		// todo: x64/x86 judge
		// System.loadLibrary("lib/opencv_java2413_x64");
		System.loadLibrary("lib/opencv_java320_x64");

	}

	public static void main(String[] args) {
		int[] myIgnore = new int[]{};
		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", myIgnore);
		//ShapeModelTrain.train("models/shape/", 0.95, false);
		ShapeModel.init("models/shape/", "V", "Z_e");
		FaceDetector.init("lbpcascade_frontalface.xml");
		// train();

		Mat pic = Imgcodecs.imread("test.jpg", Imgcodecs.IMREAD_GRAYSCALE);
		// Mat pic = MuctData.getGrayJpg(1);
		Rect faceRect = FaceDetector.searchFace(pic);

		pic.convertTo(pic, CvType.CV_32F);
		Mat patches = ImUtils.loadMat("models/patch_76");
		
		Mat myPatches = new Mat();
		Arrays.sort(myIgnore);
		
		for (int i=0;i<patches.cols();i++) {
			if (Arrays.binarySearch(myIgnore, i) >=0)
				continue;
			else
				myPatches.push_back(patches.col(i).t());
		}
		myPatches = myPatches.t();
		Mat refShape = ImUtils.loadMat("models/refShape");
		ShapeInstance shape = new ShapeInstance(faceRect.width * 0.9, 0, faceRect.x + faceRect.width / 2,
				faceRect.y + faceRect.height / 2 + faceRect.height * 0.12);
		

		RegressorSetInstance regSet = new RegressorSetInstance();
		regSet.load(myPatches, new Size(41, 41), refShape);
		regSet.init(shape.getX());
		JFrame win = new JFrame();
		for (int ttt = 0; ttt < 5000; ttt++) {
			ImUtils.startTiming();
			Mat dstPts = regSet.track(pic, new Size(21,21));
			Mat sPic = pic.clone();
			Imgproc.cvtColor(sPic, sPic, Imgproc.COLOR_GRAY2BGR);

			Mat z = ShapeModel.getZfromX(dstPts);
			ShapeModel.clamp(z, 3);
			dstPts = ShapeModel.getXfromZ(z);

			
			for (int i = 0; i < dstPts.rows() / 2; i++) {
				Imgproc.circle(sPic, new Point(dstPts.get(i * 2, 0)[0], dstPts.get(i * 2 + 1, 0)[0]), 2,
						new Scalar(0, 0, 255));
			}
			ImUtils.printTiming();

			ImUtils.imshow(win, sPic, 1);

			System.gc();
		}

	}

	public static void train() {
		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", new int[] {});
		Mat refShape = RegressorTrain.getRefShape(100, 100);
		ImUtils.saveMat(refShape, "models/refShape");
		Mat thetaSet = new Mat();
		for (int i = 0; i < MuctData.getPtsCounts(); i++) {
			System.out.println("training patch " + i + " ...");
			Mat theta = RegressorTrain.trainLinearModel(refShape, i, new Size(41, 41), new Size(21, 21), 2, 0.2,
					new LearningParams());
			thetaSet.push_back(theta.t());
			System.gc();
		}
		ImUtils.saveMat(thetaSet.t(), "models/patch_" + MuctData.getPtsCounts());
	}
}
