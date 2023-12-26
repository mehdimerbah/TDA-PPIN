#!/usr/bin/env python3 



## Import necessary unit test packages
import unittest
from pathlib import Path
import pkg_resources



_REQUIREMENTS_PATH_ = Path(__file__).parent.with_name("requirements.txt")



## Create class to test requirements with self referencing to this script and requirements in requirements.txt
class TestRequirements(unittest.TestCase):
    ## Create function to check self reference to requirements
    def test_requirements(self):
            ## Read requirements and parse into list
            requirements = pkg_resources.parse_requirements(_REQUIREMENTS_PATH_.open())
            ## Check requirements one by one, running unit tests for each
            for requirement in requirements:
                requirement = str(requirement)
                with self.subTest(requirement=requirement):
                    pkg_resources.require(requirement)