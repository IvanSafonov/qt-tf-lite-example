#include <QObject>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>

class Model : public QObject
{
public:
  Model(QObject *parent = nullptr);

  void setThreshold(float value);
  void setNumOfResults(int value);
  bool loadModel(const QString &modelFile, const QString &labelsFile);

  struct Result
  {
    int index{};
    float confidence{};
    QString label;
  };

  QVector<Result> runInference(const QString &filename);

private:
  QStringList labels;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  float threshold = 0.001f;
  int numOfResults = 5;
};
