import unittest
from unittest.mock import patch
from core.resource_manager import get_system_info

class TestResourceManager(unittest.TestCase):

    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_count')
    def test_get_system_info(self, mock_cpu_count, mock_virtual_memory):
        # Configure the mock objects
        mock_virtual_memory.return_value.total = 16 * 1024**3  # Mock 16 GB of RAM
        mock_cpu_count.return_value = 8  # Mock 8 CPU cores

        # Call the function to be tested
        system_info = get_system_info()

        # Assert the results
        self.assertEqual(system_info['ram'], 16.0)
        self.assertEqual(system_info['cpu_cores'], 8)

if __name__ == '__main__':
    unittest.main()
