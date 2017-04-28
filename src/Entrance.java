import java.util.Arrays;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import pers.season.vml.statistics.regressor.LearningParams;
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
		// train();
		VideoCapture vc = new VideoCapture();
		vc.open(0);
		// int[] myIgnore = new int[] {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
		// ,68,69,70,71,72,73,74,75};
		// int[] myIgnore = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
		// 12, 13, 14 };
		// int[] myIgnore = new int[]
		// {48,49,50,51,52,53,54,55,56,57,58,59,68,69,70,71,72,73,74,75};
		int[] myIgnore = new int[] {};
		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", myIgnore);
		ShapeModelTrain.train("models/shape/", 0.90, false);

		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		FaceDetector fd = FaceDetector.load("lbpcascade_frontalface.xml");
		Mat patches = ImUtils.loadMat("models/regressor/patch_76");
		Mat refShape = ImUtils.loadMat("models/regressor/refShape");

		Mat myPatches = new Mat();
		Mat myRefShape = new Mat();
		Arrays.sort(myIgnore);

		for (int i = 0; i < patches.cols(); i++) {
			if (Arrays.binarySearch(myIgnore, i) >= 0)
				continue;
			else {
				myPatches.push_back(patches.col(i).t());
				myRefShape.push_back(refShape.row(i * 2));
				myRefShape.push_back(refShape.row(i * 2 + 1));
			}
		}
		myPatches = myPatches.t();
		JFrame win = new JFrame();
		// Mat pic = Imgcodecs.imread("test.jpg", Imgcodecs.IMREAD_GRAYSCALE);
		// Mat pic = MuctData.getGrayJpg(2);
		while (true) {
			Mat pic = null;
			Rect faceRect = null;
			while (faceRect == null || faceRect.width < 100) {
				pic = new Mat();
				vc.read(pic);
				Imgproc.cvtColor(pic, pic, Imgproc.COLOR_BGR2GRAY);
				// ImUtils.imshow(pic);
				faceRect = fd.searchFace(pic);
				System.out.println(faceRect);
			}
			ShapeInstance shape = new ShapeInstance(sm);
			shape.setFromParams(faceRect.width * 0.9, 0, faceRect.x + faceRect.width / 2,
					faceRect.y + faceRect.height / 2 + faceRect.height * 0.12);
			RegressorSetInstance regSet = RegressorSetInstance.load(myPatches, new Size(41, 41), myRefShape);
			regSet.setCurPts(shape.getX());

			while (true) {
				pic = new Mat();
				vc.read(pic);
				Imgproc.cvtColor(pic, pic, Imgproc.COLOR_BGR2GRAY);
				pic.convertTo(pic, CvType.CV_32F);

				ImUtils.startTiming();

				Mat dstPts = regSet.track(pic, new Size(21, 21));

				Mat sPic = pic.clone();
				Imgproc.cvtColor(sPic, sPic, Imgproc.COLOR_GRAY2BGR);

				Mat z = sm.getZfromX(dstPts);
				double abnormal = Core.norm(dstPts, sm.getXfromZ(z)) / sm.getScale(z);

				if (abnormal > 0.18) {
					System.err.println(abnormal);
					// ShapeModel.clamp(z, 0);
					break;
				} else {

					if (abnormal > 0.15) {
						System.out.println(abnormal);
						sm.clamp(z, 0);
					} else {
						sm.clamp(z, 3);
					}
				}
				dstPts = sm.getXfromZ(z);

				regSet.setCurPts(dstPts);

				for (int i = 0; i < dstPts.rows() / 2; i++) {
					Imgproc.circle(sPic, new Point(dstPts.get(i * 2, 0)[0], dstPts.get(i * 2 + 1, 0)[0]), 2,
							new Scalar(0, 0, 255));
				}

				ImUtils.imshow(win, sPic, 1);
				// ImUtils.printTiming();
				System.gc();

			}
		}
	}

	public static void train() {
		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", new int[] {});
		Mat refShape = RegressorTrain.getRefShape(100, 100);
		ImUtils.saveMat(refShape, "models/regressor/refShape");
		Mat thetaSet = new Mat();
		for (int i = 0; i < MuctData.getPtsCounts(); i++) {
			System.out.println("training patch " + i + " ...");
			Mat theta = RegressorTrain.trainLinearModel(refShape, i, new Size(41, 41), new Size(21, 21), 2, 0.2,
					new LearningParams());
			thetaSet.push_back(theta.t());
			System.gc();
		}
		ImUtils.saveMat(thetaSet.t(), "models/regressor/patch_" + MuctData.getPtsCounts());
	}
}
