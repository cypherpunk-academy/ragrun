#!/usr/bin/env python3
"""
Tests for the assistants CLI commands

This module tests all the assistants CLI functionality including:
- list, create, delete, chat, list-files, add-files, remove-files, context
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

# Import the CLI module
from rag_cli import cli, assistants_group


class TestAssistantsListCommand:
    """Test the 'assistants list' command."""
    
    def setup_method(self):
        self.runner = CliRunner()
        self.mock_assistants = [
            {"name": "aurelian-i--schelling", "status": "active", "created_on": "2024-01-01"},
            {"name": "aloys-i--freud", "status": "active", "created_on": "2024-01-02"},
        ]
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_table_format(self, mock_manager_class):
        """Test listing assistants in table format."""
        mock_manager = Mock()
        mock_manager.list_assistants.return_value = self.mock_assistants
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, ['assistants', 'list'])
        
        assert result.exit_code == 0
        assert "Found 2 assistants:" in result.output
        assert "aurelian-i--schelling" in result.output
        assert "aloys-i--freud" in result.output
        assert "active" in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_json_format(self, mock_manager_class):
        """Test listing assistants in JSON format."""
        mock_manager = Mock()
        mock_manager.list_assistants.return_value = self.mock_assistants
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, ['assistants', 'list', '--output-format', 'json'])
        
        assert result.exit_code == 0
        output_data = json.loads(result.output)
        assert len(output_data) == 2
        assert output_data[0]["name"] == "aurelian-i--schelling"
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_csv_format(self, mock_manager_class):
        """Test listing assistants in CSV format."""
        mock_manager = Mock()
        mock_manager.list_assistants.return_value = self.mock_assistants
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, ['assistants', 'list', '--output-format', 'csv'])
        
        assert result.exit_code == 0
        assert "name,status,created_on" in result.output
        assert "aurelian-i--schelling,active,2024-01-01" in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_save_to_file(self, mock_manager_class):
        """Test saving assistant list to file."""
        mock_manager = Mock()
        mock_manager.list_assistants.return_value = self.mock_assistants
        mock_manager_class.return_value = mock_manager
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'assistants', 'list', 
                '--output-format', 'json', 
                '--output-file', output_file
            ])
            
            assert result.exit_code == 0
            assert f"Output saved to {output_file}" in result.output
            
            # Verify file content
            with open(output_file, 'r') as f:
                data = json.load(f)
                assert len(data) == 2
        finally:
            os.unlink(output_file)
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_no_assistants(self, mock_manager_class):
        """Test listing when no assistants exist."""
        mock_manager = Mock()
        mock_manager.list_assistants.return_value = []
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, ['assistants', 'list'])
        
        assert result.exit_code == 0
        assert "No assistants found." in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_error_handling(self, mock_manager_class):
        """Test error handling in list command."""
        mock_manager_class.side_effect = Exception("Connection failed")
        
        result = self.runner.invoke(cli, ['assistants', 'list'])
        
        assert result.exit_code == 1
        assert "Error listing assistants: Connection failed" in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_with_none_values(self, mock_manager_class):
        """Test listing assistants when some fields are None."""
        mock_assistants_with_none = [
            {"name": "test-assistant", "status": None, "created_on": "2024-01-01"},
            {"name": "another-assistant", "status": "active", "created_on": None},
            {"name": None, "status": "active", "created_on": "2024-01-02"},
        ]
        
        mock_manager = Mock()
        mock_manager.list_assistants.return_value = mock_assistants_with_none
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, ['assistants', 'list'])
        
        assert result.exit_code == 0
        assert "Found 3 assistants:" in result.output
        assert "test-assistant" in result.output
        assert "Unknown" in result.output  # Should show "Unknown" for None values


class TestAssistantsCreateCommand:
    """Test the 'assistants create' command."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    @patch('scripts.create_pinecone_assistants.OPENAI_ASSISTANT_CONFIGS')
    def test_create_with_default_instructions(self, mock_configs, mock_manager_class):
        """Test creating assistant with default instructions."""
        mock_configs.__getitem__.return_value = {
            'instructions': 'Test instructions for Idealismus'
        }
        mock_configs.__contains__.return_value = True
        
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_manager.get_or_create_assistant.return_value = mock_assistant
        mock_manager.chat_with_assistant.return_value = {
            'message': 'Hallo, ich bin ein Idealismus Assistent.'
        }
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, [
            'assistants', 'create', 
            'Test Assistant', 'Idealismus'
        ])
        
        assert result.exit_code == 0
        assert "Creating assistant 'test-assistant' for worldview 'Idealismus'" in result.output
        assert "✅ Successfully created assistant: test-assistant" in result.output
        assert "🧪 Testing assistant..." in result.output
        
        # Verify the manager was called correctly
        mock_manager.get_or_create_assistant.assert_called_once_with(
            name='test-assistant',
            worldview='Idealismus',
            instructions='Test instructions for Idealismus'
        )
    
    def test_create_dry_run(self):
        """Test creating assistant in dry-run mode."""
        result = self.runner.invoke(cli, [
            'assistants', 'create', 
            'Test Assistant', 'Idealismus',
            '--dry-run'
        ])
        
        assert result.exit_code == 0
        assert "Would create assistant:" in result.output
        assert "Name: test-assistant" in result.output
        assert "Worldview: Idealismus" in result.output
        assert "Instructions: Default worldview instructions" in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_create_with_custom_instructions(self, mock_manager_class):
        """Test creating assistant with custom instructions file."""
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_manager.get_or_create_assistant.return_value = mock_assistant
        mock_manager.chat_with_assistant.return_value = {
            'message': 'Custom response'
        }
        mock_manager_class.return_value = mock_manager
        
        # Create temporary instructions file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Custom instructions for testing")
            instructions_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'assistants', 'create',
                'Custom Assistant', 'Materialismus',
                '--instructions-file', instructions_file
            ])
            
            assert result.exit_code == 0
            assert "✅ Successfully created assistant: custom-assistant" in result.output
            
            # Verify custom instructions were used
            mock_manager.get_or_create_assistant.assert_called_once_with(
                name='custom-assistant',
                worldview='Materialismus',
                instructions='Custom instructions for testing'
            )
        finally:
            os.unlink(instructions_file)
    
    def test_create_missing_instructions_file(self):
        """Test error when instructions file doesn't exist."""
        result = self.runner.invoke(cli, [
            'assistants', 'create',
            'Test Assistant', 'Idealismus',
            '--instructions-file', '/nonexistent/file.txt'
        ])
        
        assert result.exit_code == 1
        assert "Error: Instructions file not found" in result.output
    
    def test_create_invalid_worldview(self):
        """Test error with invalid worldview."""
        result = self.runner.invoke(cli, [
            'assistants', 'create',
            'Test Assistant', 'InvalidWorldview'
        ])
        
        assert result.exit_code == 2  # Click argument validation error
        assert "Invalid value for" in result.output


class TestAssistantsDeleteCommand:
    """Test the 'assistants delete' command."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_delete_success(self, mock_manager_class):
        """Test successful assistant deletion."""
        mock_manager = Mock()
        mock_manager.delete_assistant.return_value = True
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, [
            'assistants', 'delete', 'test-assistant'
        ], input='y\n')
        
        assert result.exit_code == 0
        assert "✅ Successfully deleted assistant: test-assistant" in result.output
        mock_manager.delete_assistant.assert_called_once_with('test-assistant')
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_delete_failure(self, mock_manager_class):
        """Test failed assistant deletion."""
        mock_manager = Mock()
        mock_manager.delete_assistant.return_value = False
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, [
            'assistants', 'delete', 'test-assistant'
        ], input='y\n')
        
        assert result.exit_code == 1
        assert "❌ Failed to delete assistant: test-assistant" in result.output
    
    def test_delete_cancelled(self):
        """Test cancelling deletion."""
        result = self.runner.invoke(cli, [
            'assistants', 'delete', 'test-assistant'
        ], input='n\n')
        
        assert result.exit_code == 1
        assert "Aborted" in result.output


class TestAssistantsChatCommand:
    """Test the 'assistants chat' command."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_chat_single_message(self, mock_manager_class):
        """Test single message chat."""
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_manager.pc.assistant.Assistant.return_value = mock_assistant
        mock_manager.chat_with_assistant.return_value = {
            'message': 'Hallo! Ich bin ein philosophischer Assistent.',
            'citations': ['Source 1', 'Source 2']
        }
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, [
            'assistants', 'chat',
            'test-assistant',
            'Hello, how are you?'
        ])
        
        assert result.exit_code == 0
        assert "💬 Chatting with test-assistant..." in result.output
        assert "🤖 test-assistant:" in result.output
        assert "Hallo! Ich bin ein philosophischer Assistent." in result.output
        assert "📚 Citations: 2" in result.output
        
        # Verify the manager was called (the chat history gets updated after the call)
        assert mock_manager.chat_with_assistant.called
        call_args = mock_manager.chat_with_assistant.call_args
        assert call_args[1]['assistant'] == mock_assistant
        assert call_args[1]['message'] == 'Hello, how are you?'
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_chat_with_history_file(self, mock_manager_class):
        """Test chat with history file."""
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_manager.pc.assistant.Assistant.return_value = mock_assistant
        mock_manager.chat_with_assistant.return_value = {
            'message': 'Response with history context.',
            'citations': []
        }
        mock_manager_class.return_value = mock_manager
        
        # Create temporary history file
        history_data = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(history_data, f)
            history_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'assistants', 'chat',
                'test-assistant',
                'Follow-up question',
                '--history-file', history_file
            ])
            
            assert result.exit_code == 0
            assert "Response with history context." in result.output
            assert f"💾 Chat history saved to {history_file}" in result.output
            
            # Verify history was passed (the exact history gets updated after the call)
            assert mock_manager.chat_with_assistant.called
            call_args = mock_manager.chat_with_assistant.call_args
            assert call_args[1]['assistant'] == mock_assistant
            assert call_args[1]['message'] == 'Follow-up question'
            
            # Verify updated history was saved
            with open(history_file, 'r') as f:
                updated_history = json.load(f)
                assert len(updated_history) >= 4  # At least original 2 + new 2
        finally:
            os.unlink(history_file)
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_chat_error_handling(self, mock_manager_class):
        """Test chat error handling."""
        mock_manager_class.side_effect = Exception("Assistant not found")
        
        result = self.runner.invoke(cli, [
            'assistants', 'chat',
            'nonexistent-assistant',
            'Hello'
        ])
        
        assert result.exit_code == 1
        assert "Error chatting with assistant: Assistant not found" in result.output


class TestAssistantsListFilesCommand:
    """Test the 'assistants list-files' command."""
    
    def setup_method(self):
        self.runner = CliRunner()
        self.mock_files = [
            {"id": "file-123", "filename": "document1.txt", "status": "processed", "size": "1024"},
            {"id": "file-456", "filename": "document2.txt", "status": "processing", "size": "2048"}
        ]
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_files_table_format(self, mock_manager_class):
        """Test listing files in table format."""
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_assistant.list_files.return_value = self.mock_files
        mock_manager.pc.assistant.Assistant.return_value = mock_assistant
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, [
            'assistants', 'list-files', 'test-assistant'
        ])
        
        assert result.exit_code == 0
        assert "Files for assistant 'test-assistant':" in result.output
        assert "file-123" in result.output
        assert "document1.txt" in result.output
        assert "processed" in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_list_files_no_files(self, mock_manager_class):
        """Test listing files when no files exist."""
        mock_manager = Mock()
        mock_assistant = Mock()
        # Simulate no list_files method or empty return
        mock_assistant.list_files.side_effect = AttributeError()
        mock_manager.pc.assistant.Assistant.return_value = mock_assistant
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, [
            'assistants', 'list-files', 'test-assistant'
        ])
        
        assert result.exit_code == 0
        assert "No files found for assistant: test-assistant" in result.output


class TestAssistantsAddFilesCommand:
    """Test the 'assistants add-files' command."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_add_files_success(self, mock_manager_class):
        """Test successful file addition."""
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_manager.pc.assistant.Assistant.return_value = mock_assistant
        mock_manager.upload_documents_to_assistant.return_value = [
            {"file": "test1.txt", "status": "success"},
            {"file": "test2.txt", "status": "success"}
        ]
        mock_manager_class.return_value = mock_manager
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = os.path.join(temp_dir, "test1.txt")
            file2 = os.path.join(temp_dir, "test2.txt")
            
            with open(file1, 'w') as f:
                f.write("Test content 1")
            with open(file2, 'w') as f:
                f.write("Test content 2")
            
            result = self.runner.invoke(cli, [
                'assistants', 'add-files',
                'test-assistant',
                file1, file2,
                '--worldview', 'Idealismus'
            ])
            
            assert result.exit_code == 0
            assert "Uploading 2 files to assistant 'test-assistant'" in result.output
            # The successful count might vary due to how the upload method works
            assert "successful" in result.output
            assert "0 failed" in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_add_files_dry_run(self, mock_manager_class):
        """Test add files in dry-run mode."""
        # Mock is not needed for dry-run, but add it to avoid any imports
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = os.path.join(temp_dir, "test1.txt")
            
            with open(file1, 'w') as f:
                f.write("Test content")
            
            result = self.runner.invoke(cli, [
                'assistants', 'add-files',
                'test-assistant',
                file1,
                '--dry-run'
            ])
            
            assert result.exit_code == 0
            assert "Would upload 1 files to assistant 'test-assistant':" in result.output
            assert file1 in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_add_files_nonexistent_file(self, mock_manager_class):
        """Test error with nonexistent file."""
        # Mock is needed to prevent actual API calls
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, [
            'assistants', 'add-files',
            'test-assistant',
            '/nonexistent/file.txt'
        ])
        
        assert result.exit_code == 1
        assert "No valid files to upload." in result.output


class TestAssistantsContextCommand:
    """Test the 'assistants context' command."""
    
    def setup_method(self):
        self.runner = CliRunner()
        self.mock_context = {
            "query": "test query",
            "snippets": [
                {
                    "content": "This is relevant content from document 1",
                    "score": 0.95,
                    "metadata": {"source": "doc1.txt", "worldview": "Idealismus"}
                },
                {
                    "content": "This is relevant content from document 2", 
                    "score": 0.87,
                    "metadata": {"source": "doc2.txt", "worldview": "Idealismus"}
                }
            ]
        }
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_context_query_success(self, mock_manager_class):
        """Test successful context query."""
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_manager.pc.assistant.Assistant.return_value = mock_assistant
        mock_manager.query_with_context.return_value = self.mock_context
        mock_manager_class.return_value = mock_manager
        
        result = self.runner.invoke(cli, [
            'assistants', 'context',
            'test-assistant',
            'What is the nature of ideas?'
        ])
        
        assert result.exit_code == 0
        assert "🔍 Querying context from assistant 'test-assistant'" in result.output
        assert "Query: What is the nature of ideas?" in result.output
        assert "📄 Found 2 context snippets:" in result.output
        assert "Snippet 1 (Score: 0.950):" in result.output
        assert "This is relevant content from document 1" in result.output
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    def test_context_with_filter_and_output_file(self, mock_manager_class):
        """Test context query with worldview filter and output file."""
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_manager.pc.assistant.Assistant.return_value = mock_assistant
        mock_manager.query_with_context.return_value = self.mock_context
        mock_manager_class.return_value = mock_manager
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = f.name
        
        try:
            result = self.runner.invoke(cli, [
                'assistants', 'context',
                'test-assistant',
                'test query',
                '--top-k', '3',
                '--worldview-filter', 'Idealismus',
                '--output-file', output_file
            ])
            
            assert result.exit_code == 0
            assert f"💾 Context results saved to {output_file}" in result.output
            
            # Verify the manager was called with correct filter
            mock_manager.query_with_context.assert_called_once_with(
                assistant=mock_assistant,
                query='test query',
                metadata_filter={"worldview": "Idealismus"}
            )
            
            # Verify output file content
            with open(output_file, 'r') as f:
                data = json.load(f)
                assert data["query"] == "test query"
                assert len(data["snippets"]) == 2
        finally:
            os.unlink(output_file)


class TestAssistantsIntegration:
    """Integration tests for multiple commands working together."""
    
    def setup_method(self):
        self.runner = CliRunner()
    
    @patch('assistants.deepseek_assistant_manager.DeepSeekAssistantManager')
    @patch('scripts.create_pinecone_assistants.OPENAI_ASSISTANT_CONFIGS')
    def test_create_and_list_workflow(self, mock_configs, mock_manager_class):
        """Test creating assistant and then listing it."""
        mock_configs.__getitem__.return_value = {
            'instructions': 'Test instructions'
        }
        mock_configs.__contains__.return_value = True
        
        mock_manager = Mock()
        mock_assistant = Mock()
        mock_manager.get_or_create_assistant.return_value = mock_assistant
        mock_manager.chat_with_assistant.return_value = {'message': 'Test response'}
        
        # First call for create, second for list
        mock_manager.list_assistants.return_value = [
            {"name": "test-assistant", "status": "active", "created_on": "2024-01-01"}
        ]
        mock_manager_class.return_value = mock_manager
        
        # Create assistant
        create_result = self.runner.invoke(cli, [
            'assistants', 'create', 
            'Test Assistant', 'Idealismus'
        ])
        assert create_result.exit_code == 0
        
        # List assistants
        list_result = self.runner.invoke(cli, ['assistants', 'list'])
        assert list_result.exit_code == 0
        assert "test-assistant" in list_result.output


def run_tests():
    """Run all tests and return results."""
    print("🧪 Running RAG CLI Assistants Tests...")
    print("=" * 60)
    
    # Run pytest with verbose output
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        __file__, 
        '-v', 
        '--tb=short',
        '--color=yes'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_tests()
    exit(0 if success else 1) 