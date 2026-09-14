
const MODEL_PATH = "stroke_model_web.onnx";

let session = null;

const form = document.getElementById("predictionForm");
const resultCard = document.getElementById("resultCard");
const resultTitle = document.getElementById("resultTitle");
const riskValue = document.getElementById("riskValue");
const resultMessage = document.getElementById("resultMessage");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");

async function loadModel() {
  try {
    session = await ort.InferenceSession.create(MODEL_PATH);
    predictButton.disabled = false;
    predictButton.textContent = "Analyze Risk";
    console.log("ONNX model loaded successfully");
  } catch (error) {
    console.error("Model loading error:", error);
    predictButton.disabled = true;
    predictButton.textContent = "Model Load Failed";
    alert("Model load nahi ho saka. ONNX file ka naam aur location check karo.");
  }
}

function getNumber(id) {
  return Number(document.getElementById(id).value);
}

function getText(id) {
  return document.getElementById(id).value;
}

async function predictStrokeRisk(event) {
  event.preventDefault();

  if (!session) {
    alert("Model abhi load nahi hua.");
    return;
  }

  predictButton.disabled = true;
  predictButton.textContent = "Analyzing...";

  try {
    const inputData = {
      age: new ort.Tensor("float32", [getNumber("age")], [1, 1]),
      hypertension: new ort.Tensor(
        "float32",
        [getNumber("hypertension")],
        [1, 1]
      ),
      heart_disease: new ort.Tensor(
        "float32",
        [getNumber("heartDisease")],
        [1, 1]
      ),
      avg_glucose_level: new ort.Tensor(
        "float32",
        [getNumber("glucose")],
        [1, 1]
      ),
      bmi: new ort.Tensor("float32", [getNumber("bmi")], [1, 1]),

      gender: new ort.Tensor("string", [getText("gender")], [1, 1]),
      ever_married: new ort.Tensor("string", [getText("married")], [1, 1]),
      work_type: new ort.Tensor("string", [getText("workType")], [1, 1]),
      Residence_type: new ort.Tensor("string", [getText("residence")], [1, 1]),
      smoking_status: new ort.Tensor("string", [getText("smoking")], [1, 1])
    };

    const output = await session.run(inputData);

    console.log("Model output:", output);

    let probability = 0;

    for (const key in output) {
      const data = output[key].data;

      if (data && data.length > 0) {
        const lastValue = Number(data[data.length - 1]);

        if (lastValue >= 0 && lastValue <= 1) {
          probability = lastValue;
          break;
        }
      }
    }

    const percentage = (probability * 100).toFixed(2);

    resultCard.classList.remove("hidden");
    resultTitle.textContent = "Educational Risk Estimate";
    riskValue.textContent = `${percentage}%`;

    if (probability >= 0.5) {
      resultMessage.textContent =
        "The model estimates a higher risk pattern. This is not a diagnosis. Consult a qualified healthcare professional for proper evaluation.";
      riskValue.style.color = "#ff8a8a";
    } else {
      resultMessage.textContent =
        "The model estimates a lower risk pattern. This does not guarantee that there is no medical risk.";
      riskValue.style.color = "#45e0b8";
    }
  } catch (error) {
    console.error("Prediction error:", error);
    alert("Prediction mein error aaya. Browser console mein details check karo.");
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Analyze Risk";
  }
}

resetButton.addEventListener("click", () => {
  form.reset();
  resultCard.classList.add("hidden");
  riskValue.textContent = "--%";
  riskValue.style.color = "#45e0b8";
});

form.addEventListener("submit", predictStrokeRisk);

predictButton.disabled = true;
predictButton.textContent = "Loading Model...";

loadModel();

