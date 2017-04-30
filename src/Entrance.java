import java.util.Arrays;
import java.util.List;

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
		// train();
		VideoCapture vc = new VideoCapture();
		vc.open(0);

		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", MuctData.no_ignore);
		ShapeModelTrain.train("models/shape/", 0.90, false);

		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		FaceDetector fd = FaceDetector.load("models/lbpcascade_frontalface.xml");
		RegressorSet rs = RegressorSet.load("models/regressor/", "patch_76", "refShape", new Size(41, 41));
		RegressorSetInstance rsi = new RegressorSetInstance(rs);

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
				List<Rect> faceRectList = fd.searchFace(pic);
				faceRect = faceRectList.isEmpty() ? null : faceRectList.get(0);
				ImUtils.imshow(win, pic, 1);
			}
			ShapeInstance shape = new ShapeInstance(sm);
			shape.setFromParams(faceRect.width * 0.9, 0, faceRect.x + faceRect.width / 2,
					faceRect.y + faceRect.height / 2 + faceRect.height * 0.12);

			rsi.setCurPts(shape.getX());

			while (true) {
				pic = new Mat();
				vc.read(pic);
				Imgproc.cvtColor(pic, pic, Imgproc.COLOR_BGR2GRAY);
				pic.convertTo(pic, CvType.CV_32F);

				ImUtils.startTiming();

				Mat dstPts = rsi.track(pic, new Size(21, 21));

				Mat sPic = pic.clone();
				Imgproc.cvtColor(sPic, sPic, Imgproc.COLOR_GRAY2BGR);

				Mat z = sm.getZfromX(dstPts);
				double abnormal = Core.norm(dstPts, sm.getXfromZ(z)) / sm.getScale(z);

				if (abnormal > 0.23) {
					System.err.println(abnormal);
					// ShapeModel.clamp(z, 0);
					break;
				} else {

					if (abnormal > 0.20) {
						System.out.println(abnormal);
						sm.clamp(z, 3);
					} else {
						sm.clamp(z, 3);
					}
				}
				dstPts = sm.getXfromZ(z);

				rsi.setCurPts(dstPts);

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
