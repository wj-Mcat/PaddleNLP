# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import setuptools

description = "Paddle-Pipelines: An End to End Natural Language Proceessing Development Kit Based on PaddleNLP"


def read(file: str):
    """read the content of file"""
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, file)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return content


def read_requirements():
    content = read('requirements.txt')
    packages = content.split("\n")
    return packages


setuptools.setup(name="paddle-pipelines",
                 version=read("VERSION"),
                 author="PaddlePaddle Speech and Language Team",
                 author_email="paddlenlp@baidu.com",
                 description=description,
                 long_description=read("README.md"),
                 long_description_content_type="text/markdown",
                 url="https://github.com/PaddlePaddle/PaddleNLP",
                 packages=setuptools.find_packages(
                     where='.',
                     exclude=('examples*', 'tests*', 'docs*', 'ui*',
                              'rest_api*')),
                 setup_requires=['cython', 'numpy'],
                 install_requires=read_requirements(),
                 python_requires='>=3.7',
                 classifiers=[
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: 3.9',
                     'License :: OSI Approved :: Apache Software License',
                     'Operating System :: OS Independent',
                 ],
                 license='Apache 2.0')
