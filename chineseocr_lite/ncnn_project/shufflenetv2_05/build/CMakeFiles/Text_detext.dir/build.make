# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/build

# Include any dependencies generated for this target.
include CMakeFiles/Text_detext.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Text_detext.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Text_detext.dir/flags.make

CMakeFiles/Text_detext.dir/main.cpp.o: CMakeFiles/Text_detext.dir/flags.make
CMakeFiles/Text_detext.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Text_detext.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Text_detext.dir/main.cpp.o -c /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/main.cpp

CMakeFiles/Text_detext.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Text_detext.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/main.cpp > CMakeFiles/Text_detext.dir/main.cpp.i

CMakeFiles/Text_detext.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Text_detext.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/main.cpp -o CMakeFiles/Text_detext.dir/main.cpp.s

# Object files for target Text_detext
Text_detext_OBJECTS = \
"CMakeFiles/Text_detext.dir/main.cpp.o"

# External object files for target Text_detext
Text_detext_EXTERNAL_OBJECTS =

Text_detext: CMakeFiles/Text_detext.dir/main.cpp.o
Text_detext: CMakeFiles/Text_detext.dir/build.make
Text_detext: /usr/local/lib/libopencv_dnn.so.3.4.6
Text_detext: /usr/local/lib/libopencv_ml.so.3.4.6
Text_detext: /usr/local/lib/libopencv_objdetect.so.3.4.6
Text_detext: /usr/local/lib/libopencv_shape.so.3.4.6
Text_detext: /usr/local/lib/libopencv_stitching.so.3.4.6
Text_detext: /usr/local/lib/libopencv_superres.so.3.4.6
Text_detext: /usr/local/lib/libopencv_videostab.so.3.4.6
Text_detext: /usr/local/lib/libopencv_calib3d.so.3.4.6
Text_detext: /usr/local/lib/libopencv_features2d.so.3.4.6
Text_detext: /usr/local/lib/libopencv_flann.so.3.4.6
Text_detext: /usr/local/lib/libopencv_highgui.so.3.4.6
Text_detext: /usr/local/lib/libopencv_photo.so.3.4.6
Text_detext: /usr/local/lib/libopencv_video.so.3.4.6
Text_detext: /usr/local/lib/libopencv_videoio.so.3.4.6
Text_detext: /usr/local/lib/libopencv_imgcodecs.so.3.4.6
Text_detext: /usr/local/lib/libopencv_imgproc.so.3.4.6
Text_detext: /usr/local/lib/libopencv_core.so.3.4.6
Text_detext: CMakeFiles/Text_detext.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Text_detext"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Text_detext.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Text_detext.dir/build: Text_detext

.PHONY : CMakeFiles/Text_detext.dir/build

CMakeFiles/Text_detext.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Text_detext.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Text_detext.dir/clean

CMakeFiles/Text_detext.dir/depend:
	cd /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/build /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/build /mnt/data1/yanghuiyu/project/OCR/chinese_ocr2/ncnn_project/crnn/build/CMakeFiles/Text_detext.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Text_detext.dir/depend

