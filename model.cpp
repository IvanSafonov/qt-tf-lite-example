#include "model.h"

#include <QDebug>
#include <QFile>
#include <QImage>
#include <queue>
#include <tensorflow/lite/builtin_op_data.h>
#include <tensorflow/lite/kernels/register.h>

Model::Model(QObject *parent) : QObject(parent)
{
}

void Model::setThreshold(float value)
{
  threshold = value;
}

void Model::setNumOfResults(int value)
{
  numOfResults = value;
}

bool Model::loadModel(const QString &modelFile, const QString &labelsFile)
{
  QFile file(labelsFile);
  if (!file.open(QFile::ReadOnly))
  {
    qWarning() << "Failed to load labels " << labelsFile;
    return false;
  }

  labels.clear();
  while (!file.atEnd())
    labels.append(file.readLine().simplified());

  model = tflite::FlatBufferModel::BuildFromFile(modelFile.toUtf8());
  if (!model)
  {
    qWarning() << "Failed to mmap model " << modelFile;
    return false;
  }

  model->error_reporter();

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter)
  {
    qWarning() << "Failed to construct interpreter";
    return false;
  }

  interpreter->SetNumThreads(4);
  return true;
}

namespace
{
  template <class T>
  void resize(T *out, uint8_t *in, int image_height, int image_width,
              int image_channels, int wanted_height, int wanted_width,
              int wanted_channels, TfLiteType inputType)
  {
    int number_of_pixels = image_height * image_width * image_channels;
    std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);

    int base_index = 0;

    // two inputs: input and new_sizes
    interpreter->AddTensors(2, &base_index);
    // one output
    interpreter->AddTensors(1, &base_index);
    // set input and output tensors
    interpreter->SetInputs({0, 1});
    interpreter->SetOutputs({2});

    // set parameters of tensors
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(
        0, kTfLiteFloat32, "input",
        {1, image_height, image_width, image_channels}, quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                              quant);
    interpreter->SetTensorParametersReadWrite(
        2, kTfLiteFloat32, "output",
        {1, wanted_height, wanted_width, wanted_channels}, quant);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    const TfLiteRegistration *resize_op =
        resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
    auto *params = reinterpret_cast<TfLiteResizeBilinearParams *>(
        malloc(sizeof(TfLiteResizeBilinearParams)));
    params->align_corners = false;
    params->half_pixel_centers = false;
    interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                       nullptr);

    interpreter->AllocateTensors();

    // fill input image
    // in[] are integers, cannot do memcpy() directly
    auto input = interpreter->typed_tensor<float>(0);
    for (int i = 0; i < number_of_pixels; i++)
    {
      input[i] = in[i];
    }

    // fill new_sizes
    interpreter->typed_tensor<int>(1)[0] = wanted_height;
    interpreter->typed_tensor<int>(1)[1] = wanted_width;

    interpreter->Invoke();

    auto output = interpreter->typed_tensor<float>(2);
    auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

    for (int i = 0; i < output_number_of_pixels; i++)
    {
      switch (inputType)
      {
      case kTfLiteFloat32:
        out[i] = (output[i] - 127.5f) / 127.5f;
        break;
      case kTfLiteInt8:
        out[i] = static_cast<int8_t>(output[i] - 128);
        break;
      case kTfLiteUInt8:
        out[i] = static_cast<uint8_t>(output[i]);
        break;
      default:
        break;
      }
    }
  }

  template <class T>
  void getTopN(T *prediction, int prediction_size, size_t num_results,
               float threshold, std::vector<std::pair<float, int>> *top_results,
               TfLiteType input_type)
  {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                        std::greater<std::pair<float, int>>>
        top_result_pq;

    const long count = prediction_size; // NOLINT(runtime/int)
    float value = 0.0;

    for (int i = 0; i < count; ++i)
    {
      switch (input_type)
      {
      case kTfLiteFloat32:
        value = prediction[i];
        break;
      case kTfLiteInt8:
        value = (prediction[i] + 128) / 256.0;
        break;
      case kTfLiteUInt8:
        value = prediction[i] / 255.0;
        break;
      default:
        break;
      }
      // Only add it if it beats the threshold and has a chance at being in
      // the top N.
      if (value < threshold)
      {
        continue;
      }

      top_result_pq.push(std::pair<float, int>(value, i));

      // If at capacity, kick the smallest value out.
      if (top_result_pq.size() > num_results)
      {
        top_result_pq.pop();
      }
    }

    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty())
    {
      top_results->push_back(top_result_pq.top());
      top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
  }
} // namespace

Model::Results Model::runInference(const QString &filename)
{
  QImage image(filename);
  if (image.isNull())
  {
    qWarning() << "Failed to load image" << filename;
    return {};
  }
  return runInference(image);
}

Model::Results Model::runInference(const QImage &image)
{
  int imageWidth = image.width();
  int imageHeight = image.height();

  int imageChannels = 3;
  std::vector<uint8_t> in;
  in.reserve(imageHeight * imageWidth * imageChannels);
  for (int y{}; y < imageHeight; y++)
  {
    for (int x{}; x < imageWidth; x++)
    {
      size_t inPos = (y * imageWidth + x) * imageChannels;
      auto pixel = image.pixel(x, y);
      in[inPos] = qRed(pixel);
      in[inPos + 1] = qGreen(pixel);
      in[inPos + 2] = qBlue(pixel);
    }
  }

  int input = interpreter->inputs()[0];
  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    qWarning() << "Failed to allocate tensors!";
    return {};
  }

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray *dims = interpreter->tensor(input)->dims;
  int wantedHeight = dims->data[1];
  int wantedWidth = dims->data[2];
  int wantedChannels = dims->data[3];

  auto inputType = interpreter->tensor(input)->type;
  switch (inputType)
  {
  case kTfLiteFloat32:
    resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                  imageHeight, imageWidth, imageChannels, wantedHeight,
                  wantedWidth, wantedChannels, inputType);
    break;
  case kTfLiteInt8:
    resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(),
                   imageHeight, imageWidth, imageChannels, wantedHeight,
                   wantedWidth, wantedChannels, inputType);
    break;
  case kTfLiteUInt8:
    resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                    imageHeight, imageWidth, imageChannels, wantedHeight,
                    wantedWidth, wantedChannels, inputType);
    break;
  default:
    qWarning() << "Cannot handle input type" << interpreter->tensor(input)->type;
    return {};
  }

  if (interpreter->Invoke() != kTfLiteOk)
  {
    qWarning() << "Failed to invoke tflite!";
    return {};
  }

  std::vector<std::pair<float, int>> topResults;

  int output = interpreter->outputs()[0];
  TfLiteIntArray *outputDims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto outputSize = outputDims->data[outputDims->size - 1];
  switch (interpreter->tensor(output)->type)
  {
  case kTfLiteFloat32:
    getTopN<float>(interpreter->typed_output_tensor<float>(0), outputSize,
                   numOfResults, threshold, &topResults,
                   inputType);
    break;
  case kTfLiteInt8:
    getTopN<int8_t>(interpreter->typed_output_tensor<int8_t>(0),
                    outputSize, numOfResults, threshold,
                    &topResults, inputType);
    break;
  case kTfLiteUInt8:
    getTopN<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                     outputSize, numOfResults, threshold,
                     &topResults, inputType);
    break;
  default:
    qWarning() << "Cannot handle output type " << interpreter->tensor(output)->type;
    exit(-1);
  }

  QVector<Result> results;
  results.reserve(numOfResults);
  for (const auto &result : topResults)
    results.append({result.second, result.first, labels[result.second]});

  return results;
}
