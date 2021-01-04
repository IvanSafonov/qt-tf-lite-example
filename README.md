# Qt TensorFlow Lite example

A simple example that shows how to use TensorFlow Lite with Qt. That's a modified version of the [label_image example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image).

Tested on Ubuntu 20.04.

# How to build

* Install [Visual Studio Code](https://code.visualstudio.com/) with [Remote Containers](https://code.visualstudio.com/docs/remote/containers-tutorial) extension
* Clone this project and open in Visual Studio Code
   ```bash
   git clone https://github.com/IvanSafonov/qt-tf-lite-example.git
   code ./qt-tf-lite-example
   ```
* In the Command Palette find and click **Remote-Containers: Reopen in Container**
* In the Command Palette find and click **CMake: Build**

# Launch example

The result executable file can be found in the build subdirectory.

```bash
vscode ➜ /workspaces/qt-tf-lite-example $ ./build/qt-tf-lite-example -h
Usage: ./build/qt-tf-lite-example [options] images
Qt TensorFlow Lite example

Options:
  -h, --help             Displays this help.
  -m, --model <model>    TFLite model
  -l, --labels <labels>  Model labels file

Arguments:
  images                 Input images for classification
```

```bash
vscode ➜ /workspaces/qt-tf-lite-example $ ./build/qt-tf-lite-example ./grace_hopper.jpg
INFO: Initialized TensorFlow Lite runtime.
"./grace_hopper.jpg"
0.784314 653 "military uniform"
0.105882 907 "Windsor tie"
0.0156863 458 "bow tie"
0.0117647 466 "bulletproof vest"
```
