#include "model.h"
#include <QDebug>
#include <QCoreApplication>
#include <QCommandLineParser>

int main(int argc, char **argv)
{
  QCoreApplication app(argc, argv);

  QCommandLineParser parser;
  parser.setApplicationDescription("Qt TensorFlow Lite example");
  parser.addHelpOption();
  parser.addPositionalArgument("images", "Input images for classification");
  QCommandLineOption modelFile({"m", "model"}, "TFLite model", "model",
                               "/opt/model/mobilenet_v1_1.0_224_quant.tflite");
  parser.addOption(modelFile);
  QCommandLineOption labelsFile({"l", "labels"}, "Model labels file", "labels",
                                "/opt/model/labels_mobilenet_quant_v1_224.txt");
  parser.addOption(labelsFile);
  parser.process(app);

  Model model;
  model.setNumOfResults(5);
  model.setThreshold(0.01);

  if (!model.loadModel(parser.value(modelFile), parser.value(labelsFile)))
  {
    parser.showHelp();
    return 1;
  }

  QStringList images = parser.positionalArguments();
  if (images.isEmpty())
    images.append("../grace_hopper.jpg");

  for (const QString &image : images)
  {
    qInfo() << image;
    for (const Model::Result &result : model.runInference(image))
      qInfo() << result.confidence << result.index << result.label;
    qInfo() << "";
  }

  return 0;
}