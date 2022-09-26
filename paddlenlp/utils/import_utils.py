# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from __future__ import annotations

import sys
import os
import site
import shutil
from typing import Optional
import pip
import importlib.util


def is_faster_tokenizer_available():
    package_spec = importlib.util.find_spec("faster_tokenizer")
    return package_spec is not None and package_spec.has_location


def install_package(package_name: str,
                    version: Optional[str] = None,
                    module_name: Optional[str] = None,
                    cache_dir: Optional[str] = None):
    """install the specific version of package 

    Args:
        package_name (str): the name of package
        version (str): the version of package
        module_name (str): the imported name of package
        cache_dir (str): cache dir
    """
    module_name = module_name or package_name

    # 1. remove the existing version of package
    uninstall_package(package_name, module_name)

    # 2. install the package
    if version:
        package_name += f'=={version}'

    arguments = ['install']
    if cache_dir:
        arguments += ['-t', cache_dir]
        sys.path.insert(0, cache_dir)

    arguments += ['-i', 'https://mirror.baidu.com/pypi/simple', package_name]

    pip.main(arguments)

    # 3. add site-package to the top of package
    for site_package_dir in site.getsitepackages():
        sys.path.insert(0, site_package_dir)


def uninstall_package(package_name: str, module_name: Optional[str] = None):
    """uninstall the pacakge from site-packages

    Args:
        package_name (str): the name of package
    """
    module_name = module_name or package_name
    for site_package_dir in site.getsitepackages():
        for file in os.listdir(site_package_dir):
            package_dir = os.path.join(site_package_dir, file)
            if file.startswith(package_name) and os.path.isdir(package_dir):
                shutil.rmtree(package_dir)

    for site_package_dir in site.getsitepackages():
        while sys.path[0] == site_package_dir:
            sys.path.pop(0)

    for key in list(locals().keys()):
        if module_name in key:
            del locals()[key]

    for key in list(sys.modules.keys()):
        if module_name in key:
            del sys.modules[key]
