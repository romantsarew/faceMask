import React, { useEffect, useRef } from "react";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";
import * as tf from "@tensorflow/tfjs-core";

const FaceFrameGuide = ({
  videoRef,
  isOpen,
  onFaceStatusChange,
  onFaceDistanceChange,
  onKeypointsDetected,
  onCaptureAllowedChange,
  minFramesIn = 6,
  minFramesOut = 3,
  boxInsideRequired = true,
  percentInsideRequired = 0.95,
  ellipseMargin = 1.0,
  minEyeRatio = 0.08,
  maxEyeRatio = 0.18,
  previewFlip = true,
}) => {
  const canvasRef = useRef(null);
  const detectorRef = useRef(null);
  const animationRef = useRef(null);

  const lastGoodVisualRef = useRef(false);
  const lastDistanceStatusRef = useRef(null);
  const lastCaptureAllowedRef = useRef(false);

  const goodStreakRef = useRef(0);
  const badStreakRef = useRef(0);

  const baselineEyePxRef = useRef(null);
  const calibratingStreakRef = useRef(0);
  const CALIBRATE_MIN_STREAK = 8;
  const FAR_FACTOR = 0.85;
  const CLOSE_FACTOR = 1.2;
  const LOST_RESET_STREAK = 12;
  const lostStreakRef = useRef(0);

  useEffect(() => {
    if (!isOpen) {
      cancelAnimationFrame(animationRef.current);
      detectorRef.current = null;
      return;
    }

    let cancelled = false;

    const init = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      const detector = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: "mediapipe",
          solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh",
        }
      );

      if (!cancelled) {
        detectorRef.current = detector;
        startDetectLoop();
      }
    };

    init();

    return () => {
      cancelled = true;
      cancelAnimationFrame(animationRef.current);
      detectorRef.current = null;
    };
  }, [isOpen]);

  // помощники
  const isInsideEllipse = (x, y, cx, cy, rx, ry) =>
    (x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2 <= 1;

  const getBoundingBox = (keypoints) => {
    const xs = keypoints.map((p) => p.x);
    const ys = keypoints.map((p) => p.y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
    };
  };

  const isBoundingBoxInsideEllipse = (keypoints, cx, cy, rx, ry) => {
    const b = getBoundingBox(keypoints);
    const corners = [
      { x: b.minX, y: b.minY },
      { x: b.maxX, y: b.minY },
      { x: b.minX, y: b.maxY },
      { x: b.maxX, y: b.maxY },
    ];
    return corners.every(({ x, y }) => isInsideEllipse(x, y, cx, cy, rx, ry));
  };

  const fractionInsideEllipse = (keypoints, cx, cy, rx, ry) => {
    let inside = 0;
    for (let i = 0; i < keypoints.length; i++) {
      const { x, y } = keypoints[i];
      if (isInsideEllipse(x, y, cx, cy, rx, ry)) inside += 1;
    }
    return inside / keypoints.length;
  };

  const getDistance = (p1, p2) => {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.hypot(dx, dy);
  };

  const startDetectLoop = () => {
    const detect = async () => {
      if (!detectorRef.current || !videoRef.current || !canvasRef.current) {
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video.videoWidth || !video.videoHeight) {
        animationRef.current = requestAnimationFrame(detect);
        return;
      }

      if (
        canvas.width !== video.videoWidth ||
        canvas.height !== video.videoHeight
      ) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        animationRef.current = requestAnimationFrame(detect);
        return;
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2.5;
      const radiusX = canvas.width * 0.3 * ellipseMargin;
      const radiusY = canvas.height * 0.31 * ellipseMargin;

      let isGoodNow = false;
      let captureAllowed = lastCaptureAllowedRef.current;

      const predictions = await detectorRef.current.estimateFaces(video);

      if (predictions.length > 0) {
        const face = predictions[0];
        const keypoints = face.keypoints;

        // 1) Дистанция (приоритет)
        const leftEye = keypoints[33];
        const rightEye = keypoints[263];
        const eyeDistance = getDistance(leftEye, rightEye);

        const denom = 2 * Math.min(radiusX, radiusY);
        const relativeEyeDistance = eyeDistance / Math.max(denom, 1);

        const boxOk = !boxInsideRequired
          ? true
          : isBoundingBoxInsideEllipse(
              keypoints,
              centerX,
              centerY,
              radiusX,
              radiusY
            );
        const fracOk =
          fractionInsideEllipse(
            keypoints,
            centerX,
            centerY,
            radiusX,
            radiusY
          ) >= percentInsideRequired;
        const confidenceOk =
          face?.faceInViewConfidence == null
            ? true
            : face.faceInViewConfidence > 0.95;

        if (boxOk && fracOk && confidenceOk) {
          calibratingStreakRef.current += 1;
          lostStreakRef.current = 0;
          if (
            baselineEyePxRef.current == null &&
            calibratingStreakRef.current >= CALIBRATE_MIN_STREAK
          ) {
            baselineEyePxRef.current = eyeDistance;
          }
        } else {
          calibratingStreakRef.current = 0;
        }

        let distanceStatus = "ok";
        const baseline = baselineEyePxRef.current;
        if (baseline != null) {
          if (eyeDistance < baseline * FAR_FACTOR) distanceStatus = "tooFar";
          else if (eyeDistance > baseline * CLOSE_FACTOR)
            distanceStatus = "tooClose";
        } else {
          if (relativeEyeDistance < minEyeRatio) distanceStatus = "tooFar";
          else if (relativeEyeDistance > maxEyeRatio)
            distanceStatus = "tooClose";
        }

        if (distanceStatus !== lastDistanceStatusRef.current) {
          onFaceDistanceChange?.(distanceStatus);
          lastDistanceStatusRef.current = distanceStatus;
        }

        if (distanceStatus !== "ok") {
          isGoodNow = false;
          goodStreakRef.current = 0;
          badStreakRef.current += 1;
          captureAllowed = false;
          onKeypointsDetected?.(keypoints);
        } else {
          isGoodNow = boxOk && fracOk && confidenceOk;

          if (isGoodNow) {
            goodStreakRef.current += 1;
            badStreakRef.current = 0;
          } else {
            badStreakRef.current += 1;
            goodStreakRef.current = 0;
          }

          captureAllowed =
            goodStreakRef.current >= minFramesIn
              ? true
              : badStreakRef.current >= minFramesOut
              ? false
              : lastCaptureAllowedRef.current;

          onKeypointsDetected?.(keypoints);
        }
      } else {
        lostStreakRef.current += 1;
        if (lostStreakRef.current >= LOST_RESET_STREAK) {
          baselineEyePxRef.current = null;
          calibratingStreakRef.current = 0;
        }
        if (lastDistanceStatusRef.current !== "tooFar") {
          onFaceDistanceChange?.("tooFar");
          lastDistanceStatusRef.current = "tooFar";
        }
        onKeypointsDetected?.(null);
        isGoodNow = false;
        goodStreakRef.current = 0;
        badStreakRef.current += 1;
        captureAllowed = false;
      }

      if (captureAllowed !== lastCaptureAllowedRef.current) {
        onCaptureAllowedChange?.(captureAllowed);
        lastCaptureAllowedRef.current = captureAllowed;
      }
      if (isGoodNow !== lastGoodVisualRef.current) {
        onFaceStatusChange?.(isGoodNow);
        lastGoodVisualRef.current = isGoodNow;
      }

      // ---- отрисовка ----
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.globalCompositeOperation = "destination-out";
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();

      ctx.strokeStyle = isGoodNow ? "rgba(0,255,0,0.8)" : "rgba(255,0,0,0.8)";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);
      ctx.stroke();

      animationRef.current = requestAnimationFrame(detect);
    };

    const loop = async () => {
      await detect();
      const v = videoRef.current;
      if (v && "requestVideoFrameCallback" in HTMLVideoElement.prototype) {
        v.requestVideoFrameCallback(loop);
      } else {
        animationRef.current = requestAnimationFrame(loop);
      }
    };
    loop();
  };

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        transform: "scaleX(-1)",
        transformOrigin: "center",
      }}
    />
  );
};

export default FaceFrameGuide;
