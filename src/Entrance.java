import java.util.Arrays;
import java.util.List;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
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

	static int fps = 0, frameCount = 0;

	public static void main(String[] args) {
		new Thread(new Runnable() {
			@Override
			public void run() {
				try {
					while (true) {
						fps = frameCount;
						frameCount = 0;
						Thread.sleep(1000);
					}
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}).start();
		
		// train();
		VideoCapture vc = new VideoCapture();
		vc.open(0);

		MuctData.init("e:/muct/jpg", "e:/muct/muct76-opencv.csv", MuctData.no_ignore);
		// ShapeModelTrain.train("models/shape/", 0.90, false);

		ShapeModel sm = ShapeModel.load("models/shape/", "V", "Z_e");
		// ShapeModelTrain.visualize(sm);

		FaceDetector fd = FaceDetector.load("models/lbpcascade_frontalface.xml");
		RegressorSet rs = RegressorSet.load("models/regressor/", "patch_76_size61", "refShape", new Size(61, 61));
		RegressorSetInstance rsi = new RegressorSetInstance(rs);

		JFrame win = new JFrame();

		while (true) {
			Mat pic = null;
			Rect faceRect = null;

			// search face
			while (faceRect == null || faceRect.width < 100) {
				pic = new Mat();
				vc.read(pic);
				Imgproc.cvtColor(pic, pic, Imgproc.COLOR_BGR2GRAY);
				// ImUtils.imshow(pic);
				Rect[] faceRectList = fd.searchFace(pic);
				faceRect = faceRectList.length == 0 ? null : faceRectList[0];
				ImUtils.imshow(win, pic, 1);
			}

			// initiate points position
			ShapeInstance shape = new ShapeInstance(sm);
			shape.setFromParams(faceRect.width * 0.9, 0, faceRect.x + faceRect.width / 2,
					faceRect.y + faceRect.height / 2 + faceRect.height * 0.12);
			rsi.setCurPts(shape.getX());

			while (true) {
				pic = new Mat();
				vc.read(pic);
				Imgproc.cvtColor(pic, pic, Imgproc.COLOR_BGR2GRAY);

				pic.convertTo(pic, CvType.CV_32F);

				Mat sPic = pic.clone();
				ImUtils.startTiming();

				// track the new frame
				Mat dstPts = rsi.track(pic, new Size(21, 21));

				Imgproc.cvtColor(sPic, sPic, Imgproc.COLOR_GRAY2BGR);

				// clamp the shape (using shape model)
				Mat z = sm.getZfromX(dstPts);
				sm.clamp(z, 3);
				RotatedRect roRect = sm.getLocation(z);
				ImUtils.drawRotatedRect(sPic, roRect, 2);
				Mat dspPtsClamped = sm.getXfromZ(z);
				// evaluate abnormal ranking
				double abnormal = Core.norm(dstPts, dspPtsClamped) / sm.getScale(z);
				if (abnormal > 0.20) {
					System.err.println(abnormal);
					break;
				} else {

				}

				// draw points
				for (int i = 0; i < dspPtsClamped.rows() / 2; i++) {
					Imgproc.circle(sPic, new Point(dspPtsClamped.get(i * 2, 0)[0], dspPtsClamped.get(i * 2 + 1, 0)[0]),
							1, new Scalar(0, 0, 255), 2);
				}

				// conservative set next starting point positions
				Mat nxtPts = Mat.zeros(z.size(), z.type());
				int reserveParams = 4 + 4;
				z.rowRange(0, reserveParams).copyTo(nxtPts.rowRange(0, reserveParams));
				rsi.setCurPts(sm.getXfromZ(nxtPts));
				Imgproc.putText(sPic, "fps : " + fps, new Point(20, pic.height() - 20), Core.FONT_HERSHEY_PLAIN, 1,
						new Scalar(0, 0, 255), 2);
				ImUtils.imshow(win, sPic, 1);
				
				frameCount++;
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
			Mat theta = RegressorTrain.trainLinearModel(refShape, i, new Size(61, 61), new Size(61, 61), 3, 0.2,
					new LearningParams());
			thetaSet.push_back(theta.t());
			System.gc();
		}
		ImUtils.saveMat(thetaSet.t(), "models/regressor/patch_" + MuctData.getPtsCounts());
	}
}
