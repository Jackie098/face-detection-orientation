import "./style.css";

import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";
import cv from "@techstark/opencv-js";

let faceLandmarker: FaceLandmarker | null = null;

// document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
//   <div>
//     <a href="https://vitejs.dev" target="_blank">
//       <img src="${viteLogo}" class="logo" alt="Vite logo" />
//     </a>
//     <a href="https://www.typescriptlang.org/" target="_blank">
//       <img src="${typescriptLogo}" class="logo vanilla" alt="TypeScript logo" />
//     </a>
//     <h1>Vite + TypeScript</h1>
//     <div class="card">
//       <button id="counter" type="button"></button>
//     </div>
//     <p class="read-the-docs">
//       Click on the Vite and TypeScript logos to learn more
//     </p>
//   </div>
// `;

// setupCounter(document.querySelector<HTMLButtonElement>("#counter")!);
// Image EXAMPLE
const inputImageElement = document.getElementById("inputFile");
const canvasElement = document.getElementById("output") as HTMLCanvasElement;
const canvasCtx = canvasElement.getContext("2d");

async function loadImageToCanvas(file: File): Promise<HTMLImageElement> {
  const img = new Image();
  const objectURL = URL.createObjectURL(file);
  img.src = objectURL;

  return new Promise((resolve) => {
    img.onload = () => {
      canvasElement.width = img.width;
      canvasElement.height = img.height;
      canvasCtx!.drawImage(img, 0, 0);
      resolve(img);
    };
  });
}

async function loadFaceLandmarker(runningMode: "IMAGE" | "VIDEO") {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    runningMode: runningMode,
    numFaces: 1,
  });
}

function drawLandmarksToCanvas(
  faceLandmarks: NormalizedLandmark[][],
  drawingUtils: DrawingUtils
) {
  for (const landmarks of faceLandmarks) {
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_TESSELATION,
      { color: "#C0C0C070", lineWidth: 1 }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
      { color: "#FF3030" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
      { color: "#FF3030" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
      { color: "#30FF30" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
      { color: "#30FF30" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
      { color: "#E0E0E0" }
    );
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
      color: "#E0E0E0",
    });
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
      { color: "#FF3030" }
    );
    drawingUtils.drawConnectors(
      landmarks,
      FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
      { color: "#30FF30" }
    );
  }
}

function calculateHeadDepth(faceLandmarks: NormalizedLandmark[][]) {
  const nose = faceLandmarks[0][1];

  // Calcular a distância aproximada com base no valor z
  const zDistance = Math.abs(nose.z * 100); // Multiplicado por 100 para uma escala aproximada

  return { zDistance };
}

function calculateHeadOrientation(
  landmarks: NormalizedLandmark[],
  { width, height }: { width: number; height: number }
): {
  yaw: number;
  pitch: number;
  roll: number;
} {
  const noseTip = landmarks[1]; // Landmarks para a ponta do nariz
  const rightEye = landmarks[33]; // Landmarks para o olho direito
  const leftEye = landmarks[263]; // Landmarks para o olho esquerdo
  const chin = landmarks[152]; // Landmarks para o queixo

  const face2d = [];
  var points = [1, 33, 263, 61, 291, 199];
  var pointsObj = [
    0,
    -1.126865,
    7.475604, // nose 1
    -4.445859,
    2.663991,
    3.173422, //left eye corner 33
    4.445859,
    2.663991,
    3.173422, //right eye corner 263
    -2.456206,
    -4.342621,
    4.283884, // left mouth corner 61
    2.456206,
    -4.342621,
    4.283884, // right mouth corner 291
    0,
    -9.403378,
    4.264492,
  ];

  let yaw = 0,
    pitch = 0,
    roll = 0;

  var x = 0,
    y = 0,
    z = 0;

  const normalizedFocaleY = 1.28;
  const focalLength = height * normalizedFocaleY;
  const s = 0;
  const cx = width / 2;
  const cy = height / 2;

  const camMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
    focalLength,
    s,
    cx,
    0,
    focalLength,
    cy,
    0,
    0,
    1,
  ]);
  console.log("🚀 ~ camMatrix:", camMatrix);

  var k1 = 0.1318020374;
  var k2 = -0.1550007612;
  var p1 = -0.0071350401;
  var p2 = -0.0096747708;
  var dist_matrix = cv.matFromArray(4, 1, cv.CV_64FC1, [k1, k2, p1, p2]);

  var message = "";

  if (landmarks) {
    // for (const landmark of landmarks) {
    // drawingUtils.drawConnectors(
    //   canvasCtx,
    //   landmark,
    //   mpFaceMesh.FACEMESH_TESSELATION,
    //   { color: "#C0C0C070", lineWidth: 1 }
    // );
    // }

    for (const point of points) {
      var point0 = landmarks[point];

      //console.log("landmarks : " + landmarks.landmark.data64F);

      // drawingUtils.drawLandmarks(canvasCtx, [point0], { color: "#FFFFFF" }); // expects normalized landmark

      var x = point0.x * width;
      var y = point0.y * height;
      //var z = point0.z;

      // Get the 2D Coordinates
      face2d.push(x);
      face2d.push(y);
    }
  }

  if (face2d.length > 0) {
    const rvec = new cv.Mat();
    const tvec = new cv.Mat();

    const numRows = points.length;
    console.log("🚀 ~ numRows:", numRows);
    console.log("🚀 ~ face2d:", face2d);
    const imagePoints = cv.matFromArray(numRows, 2, cv.CV_64FC1, face2d);

    var modelPointsObj = cv.matFromArray(6, 3, cv.CV_64FC1, pointsObj);
    console.log("🚀 ~ modelPointsObj:", modelPointsObj);

    var success = cv.solvePnP(
      modelPointsObj, //modelPoints,
      imagePoints,
      camMatrix,
      dist_matrix,
      rvec, // Output rotation vector
      tvec,
      false, //  uses the provided rvec and tvec values as initial approximations
      cv.SOLVEPNP_ITERATIVE //SOLVEPNP_EPNP //SOLVEPNP_ITERATIVE (default but pose seems unstable)
    );

    if (success) {
      var rmat = cv.Mat.zeros(3, 3, cv.CV_64FC1);
      const jaco = new cv.Mat();

      console.log("rvec", rvec.data64F[0], rvec.data64F[1], rvec.data64F[2]);
      console.log("tvec", tvec.data64F[0], tvec.data64F[1], tvec.data64F[2]);

      // Get rotational matrix rmat
      cv.Rodrigues(rvec, rmat, jaco); // jacobian	Optional output Jacobian matrix

      var sy = Math.sqrt(
        rmat.data64F[0] * rmat.data64F[0] + rmat.data64F[3] * rmat.data64F[3]
      );

      var singular = sy < 1e-6;

      // we need decomposeProjectionMatrix

      if (!singular) {
        //console.log("!singular");
        x = Math.atan2(rmat.data64F[7], rmat.data64F[8]);
        y = Math.atan2(-rmat.data64F[6], sy);
        z = Math.atan2(rmat.data64F[3], rmat.data64F[0]);
      } else {
        console.log("singular");
        x = Math.atan2(-rmat.data64F[5], rmat.data64F[4]);
        //  x = Math.atan2(rmat.data64F[1], rmat.data64F[2]);
        y = Math.atan2(-rmat.data64F[6], sy);
        z = 0;
      }

      roll = z;
      pitch = x;
      yaw = y;

      var worldPoints = cv.matFromArray(9, 3, cv.CV_64FC1, [
        modelPointsObj.data64F[0] + 3,
        modelPointsObj.data64F[1],
        modelPointsObj.data64F[2], // x axis
        modelPointsObj.data64F[0],
        modelPointsObj.data64F[1] + 3,
        modelPointsObj.data64F[2], // y axis
        modelPointsObj.data64F[0],
        modelPointsObj.data64F[1],
        modelPointsObj.data64F[2] - 3, // z axis
        modelPointsObj.data64F[0],
        modelPointsObj.data64F[1],
        modelPointsObj.data64F[2], //
        modelPointsObj.data64F[3],
        modelPointsObj.data64F[4],
        modelPointsObj.data64F[5], //
        modelPointsObj.data64F[6],
        modelPointsObj.data64F[7],
        modelPointsObj.data64F[8], //
        modelPointsObj.data64F[9],
        modelPointsObj.data64F[10],
        modelPointsObj.data64F[11], //
        modelPointsObj.data64F[12],
        modelPointsObj.data64F[13],
        modelPointsObj.data64F[14], //
        modelPointsObj.data64F[15],
        modelPointsObj.data64F[16],
        modelPointsObj.data64F[17], //
      ]);

      //console.log("worldPoints : " + worldPoints.data64F);

      var imagePointsProjected = new cv.Mat(
        { width: 9, height: 2 },
        cv.CV_64FC1
      );
      cv.projectPoints(
        worldPoints, // TODO object points that never change !
        rvec,
        tvec,
        camMatrix,
        dist_matrix,
        imagePointsProjected,
        jaco
      );

      // Draw pose

      // canvasCtx.lineWidth = 5;

      var scaleX = canvasElement.width / width;
      var scaleY = canvasElement.height / height;

      // canvasCtx.strokeStyle = "red";
      // canvasCtx.beginPath();
      // canvasCtx.moveTo(
      //   imagePointsProjected.data64F[6] * scaleX,
      //   imagePointsProjected.data64F[7] * scaleX
      // );
      // canvasCtx.lineTo(
      //   imagePointsProjected.data64F[0] * scaleX,
      //   imagePointsProjected.data64F[1] * scaleY
      // );
      // canvasCtx.closePath();
      // canvasCtx.stroke();

      // canvasCtx.strokeStyle = "green";
      // canvasCtx.beginPath();
      // canvasCtx.moveTo(
      //   imagePointsProjected.data64F[6] * scaleX,
      //   imagePointsProjected.data64F[7] * scaleX
      // );
      // canvasCtx.lineTo(
      //   imagePointsProjected.data64F[2] * scaleX,
      //   imagePointsProjected.data64F[3] * scaleY
      // );
      // canvasCtx.closePath();
      // canvasCtx.stroke();

      // canvasCtx.strokeStyle = "blue";
      // canvasCtx.beginPath();
      // canvasCtx.moveTo(
      //   imagePointsProjected.data64F[6] * scaleX,
      //   imagePointsProjected.data64F[7] * scaleX
      // );
      // canvasCtx.lineTo(
      //   imagePointsProjected.data64F[4] * scaleX,
      //   imagePointsProjected.data64F[5] * scaleY
      // );
      // canvasCtx.closePath();
      // canvasCtx.stroke();

      // // https://developer.mozilla.org/en-US/docs/Web/CSS/named-color
      // canvasCtx.fillStyle = "aqua";

      // for (var i = 6; i <= 6 + 6 * 2; i += 2) {
      //   canvasCtx.rect(
      //     imagePointsProjected.data64F[i] * scaleX - 5,
      //     imagePointsProjected.data64F[i + 1] * scaleY - 5,
      //     10,
      //     10
      //   );
      //   canvasCtx.fill();
      // }

      jaco.delete();
      imagePointsProjected.delete();
    }

    rvec.delete();
    tvec.delete();
  }

  return { yaw, pitch, roll };
}

function displayOrientationResultMessage(
  yaw: number,
  pitch: number,
  roll: number
) {
  //  @ts-ignore
  const msgRoll =
    "roll: " +
    (180.0 * (roll / Math.PI)).toFixed(2) +
    ` - ${
      //  @ts-ignore
      (180.0 * (roll / Math.PI)).toFixed(2) < 10 &&
      //  @ts-ignore
      (180.0 * (roll / Math.PI)).toFixed(2) > -10
        ? "de frente"
        : "rolando"
    }`;
  const msgPitch =
    "pitch: " +
    (180.0 * (pitch / Math.PI)).toFixed(2) +
    ` - ${
      //  @ts-ignore
      (180.0 * (pitch / Math.PI)).toFixed(2) > -170 &&
      //  @ts-ignore
      (180.0 * (pitch / Math.PI)).toFixed(2) < -149
        ? "de frente"
        : "pra cima ou baixo"
    }`;
  const msgYaw =
    "yaw: " +
    (180.0 * (yaw / Math.PI)).toFixed(2) +
    ` - ${
      //  @ts-ignore
      (180.0 * (yaw / Math.PI)).toFixed(2) > -3 &&
      //  @ts-ignore
      (180.0 * (yaw / Math.PI)).toFixed(2) < 3
        ? "de frente"
        : "para o lado"
    }`;

  return { msgRoll, msgPitch, msgYaw };
}

inputImageElement?.addEventListener("change", async (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];

  if (file) {
    const img = await loadImageToCanvas(file);
    await loadFaceLandmarker("IMAGE");

    const faceLandmarkerResult = await faceLandmarker!.detect(img);

    const drawingUtils = new DrawingUtils(canvasCtx!);
    drawLandmarksToCanvas(faceLandmarkerResult.faceLandmarks, drawingUtils);

    const { zDistance } = calculateHeadDepth(
      faceLandmarkerResult.faceLandmarks
    );
    // @ts-ignore
    document.getElementById(
      "distance"
    ).innerText = `Distância estimada do nariz: ${zDistance.toFixed(2)} in z`;

    const landmarks = faceLandmarkerResult.faceLandmarks[0];
    // @ts-ignore
    const { yaw, pitch, roll } = calculateHeadOrientation(landmarks, {
      width: img.width,
      height: img.height,
    });

    // canvasCtx.fillStyle = "black";
    // canvasCtx.font = "bold 30px Arial";
    // canvasCtx.fillText(
    //   "roll: " + (180.0 * (roll / Math.PI)).toFixed(2) + ` - ${(180.0 * (roll / Math.PI)).toFixed(2) < 10 && (180.0 * (roll / Math.PI)).toFixed(2) > -10 ? "de frente" : "rolando"}`,
    //   //"roll: " + roll.toFixed(2),
    //   width * 0.8,
    //   50
    // );
    // canvasCtx.fillText(
    //   "pitch: " + (180.0 * (pitch / Math.PI)).toFixed(2) + ` - ${(180.0 * (pitch / Math.PI)).toFixed(2) > -170 && (180.0 * (pitch / Math.PI)).toFixed(2) < -149 ? "de frente" : "pra cima ou baixo"}`,
    //   //"pitch: " + pitch.toFixed(2),
    //   width * 0.8,
    //   100
    // );
    // canvasCtx.fillText(
    //   "yaw: " +
    //     (180.0 * (yaw / Math.PI)).toFixed(2) +
    //     ` - ${
    //       (180.0 * (yaw / Math.PI)).toFixed(2) > -3 &&
    //       (180.0 * (yaw / Math.PI)).toFixed(2) < 3
    //         ? "de frente"
    //         : "para o lado"
    //     }`,
    //   //"yaw: " + yaw.toFixed(3),
    //   width * 0.8,
    //   150
    // );

    console.log(
      "pose %f %f %f",
      (180.0 * (roll / Math.PI)).toFixed(2),
      (180.0 * (pitch / Math.PI)).toFixed(2),
      (180.0 * (yaw / Math.PI)).toFixed(2)
    );

    const { msgRoll, msgPitch, msgYaw } = displayOrientationResultMessage(
      yaw,
      pitch,
      roll
    );

    document.getElementById(
      "orientation"
    )!.innerText = `Orientação do rosto: ${msgRoll} | ${msgPitch} | ${msgYaw}`;
  }
});
