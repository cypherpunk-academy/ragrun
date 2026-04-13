#!/usr/bin/env python3
"""
Script to add gedankenfehler-umkehren results to the 12_weltanschauungen database.
Handles rank management and user interaction for prioritization.
"""

import os
import sys
import json
import uuid
from datetime import datetime
from pymongo import MongoClient
from typing import Dict, List, Any, Optional

class GedankenfehlerUmkehrenIntegrator:
    """Handles integration of gedankenfehler-umkehren results into the database."""
    
    def __init__(self, mongodb_uri: str = None):
        """Initialize the integrator with database connection."""
        self.mongodb_uri = mongodb_uri or os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
        self.client = None
        self.db = None
        self.gedanken_collection = None
        self.autoren_collection = None
        
    def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client['12_weltanschauungen']
            self.gedanken_collection = self.db['gedanken']
            self.autoren_collection = self.db['autoren']
            print("✅ Connected to MongoDB")
            return True
        except Exception as e:
            print(f"❌ Error connecting to MongoDB: {e}")
            return False
    
    def get_existing_entries(self, weltanschauung: str, nummer: int) -> List[Dict]:
        """Get existing entries for a specific weltanschauung and nummer combination."""
        query = {'weltanschauung': weltanschauung, 'nummer': nummer}
        entries = list(self.gedanken_collection.find(query).sort('rank', 1))
        return entries
    
    def get_next_rank(self, weltanschauung: str, nummer: int) -> int:
        """Get the next available rank for a weltanschauung/nummer combination."""
        existing_entries = self.get_existing_entries(weltanschauung, nummer)
        if not existing_entries:
            return 1
        
        max_rank = max(entry.get('rank', 0) for entry in existing_entries)
        return max_rank + 1
    
    def get_author_info(self, author_name: str) -> Optional[Dict]:
        """Get author information from autoren collection."""
        return self.autoren_collection.find_one({'name': author_name})
    
    def display_existing_entries(self, weltanschauung: str, nummer: int):
        """Display existing entries for user review."""
        entries = self.get_existing_entries(weltanschauung, nummer)
        
        if not entries:
            print(f"   ℹ️  No existing entries for {weltanschauung} #{nummer}")
            return
        
        print(f"\n📋 EXISTING ENTRIES for {weltanschauung} #{nummer}:")
        print("-" * 60)
        
        for entry in entries:
            print(f"   Rank {entry.get('rank', 'N/A')}: {entry.get('autor', 'Unknown Author')}")
            print(f"   Created: {entry.get('created_at', 'N/A')}")
            print(f"   Model: {entry.get('model', 'N/A')}")
            
            # Show brief preview of the gedanke
            gedanke = entry.get('gedanke', '')
            preview = gedanke[:100] + "..." if len(gedanke) > 100 else gedanke
            print(f"   Preview: {preview}")
            print("-" * 60)
    
    def ask_user_for_rank_priority(self, weltanschauung: str, nummer: int, next_rank: int) -> int:
        """Ask user how they want to prioritize the new entry."""
        self.display_existing_entries(weltanschauung, nummer)
        
        print(f"\n❓ RANK PRIORITIZATION for {weltanschauung} #{nummer}:")
        print(f"   • Default next rank would be: {next_rank}")
        print(f"   • You can:")
        print(f"     1) Accept default rank ({next_rank})")
        print(f"     2) Specify a different rank (will adjust existing entries)")
        print(f"     3) Skip this entry")
        
        while True:
            choice = input(f"\nChoose option (1/2/3): ").strip()
            
            if choice == '1':
                return next_rank
            elif choice == '2':
                try:
                    custom_rank = int(input(f"Enter desired rank (1-{next_rank}): ").strip())
                    if 1 <= custom_rank <= next_rank:
                        return custom_rank
                    else:
                        print(f"❌ Rank must be between 1 and {next_rank}")
                except ValueError:
                    print("❌ Please enter a valid number")
            elif choice == '3':
                return -1  # Skip indicator
            else:
                print("❌ Please choose 1, 2, or 3")
    
    def adjust_existing_ranks(self, weltanschauung: str, nummer: int, new_rank: int):
        """Adjust ranks of existing entries when inserting at a specific position."""
        # Get all entries with rank >= new_rank
        query = {
            'weltanschauung': weltanschauung, 
            'nummer': nummer, 
            'rank': {'$gte': new_rank}
        }
        
        entries_to_adjust = list(self.gedanken_collection.find(query))
        
        if entries_to_adjust:
            print(f"   🔄 Adjusting ranks for {len(entries_to_adjust)} existing entries...")
            
            # Update ranks in reverse order to avoid conflicts
            for entry in sorted(entries_to_adjust, key=lambda x: x.get('rank', 0), reverse=True):
                old_rank = entry.get('rank', 0)
                new_adjusted_rank = old_rank + 1
                
                self.gedanken_collection.update_one(
                    {'_id': entry['_id']},
                    {'$set': {'rank': new_adjusted_rank}}
                )
                print(f"     • {entry.get('autor', 'Unknown')}: rank {old_rank} → {new_adjusted_rank}")
    
    def validate_document(self, document: Dict) -> tuple[bool, List[str]]:
        """Validate a document before insertion."""
        errors = []
        
        # Required fields
        required_fields = [
            'autor', 'weltanschauung', 'nummer', 'ausgangsgedanke',
            'gedanke', 'gedanke_einfach', 'gedanke_kurz'
        ]
        
        for field in required_fields:
            if not document.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate nummer range
        nummer = document.get('nummer')
        if nummer and not (1 <= nummer <= 43):
            errors.append(f"Nummer {nummer} is outside valid range (1-43)")
        
        # Validate author exists
        autor = document.get('autor')
        if autor and not self.get_author_info(autor):
            errors.append(f"Author '{autor}' not found in autoren collection")
        
        # Validate weltanschauung
        valid_weltanschauungen = self.gedanken_collection.distinct('weltanschauung')
        weltanschauung = document.get('weltanschauung')
        if weltanschauung and weltanschauung not in valid_weltanschauungen:
            errors.append(f"Unknown weltanschauung: {weltanschauung}")
        
        return len(errors) == 0, errors
    
    def prepare_document(self, raw_data: Dict, target_rank: int = None) -> Dict:
        """Prepare a document for database insertion."""
        # Get author info
        autor_info = self.get_author_info(raw_data.get('autor', ''))
        autor_id = autor_info.get('id') if autor_info else 'unknown'
        
        # Calculate rank if not provided
        if target_rank is None:
            target_rank = self.get_next_rank(
                raw_data.get('weltanschauung', ''),
                raw_data.get('nummer', 0)
            )
        
        document = {
            'autor': raw_data.get('autor', ''),
            'autorId': autor_id,
            'weltanschauung': raw_data.get('weltanschauung', ''),
            'created_at': raw_data.get('created_at', datetime.now()),
            'ausgangsgedanke': raw_data.get('ausgangsgedanke', ''),
            'ausgangsgedanke_in_weltanschauung': raw_data.get('ausgangsgedanke_in_weltanschauung', ''),
            'id': raw_data.get('id', str(uuid.uuid4())),
            'gedanke': raw_data.get('gedanke', ''),
            'gedanke_einfach': raw_data.get('gedanke_einfach', ''),
            'gedanke_kurz': raw_data.get('gedanke_kurz', ''),
            'nummer': raw_data.get('nummer', 0),
            'model': raw_data.get('model', 'gedankenfehler-umkehren'),
            'rank': target_rank
        }
        
        return document
    
    def add_single_entry(self, raw_data: Dict, interactive: bool = True) -> bool:
        """Add a single gedankenfehler-umkehren entry to the database."""
        try:
            weltanschauung = raw_data.get('weltanschauung', '')
            nummer = raw_data.get('nummer', 0)
            
            print(f"\n🔄 Processing {weltanschauung} #{nummer}...")
            
            # Determine rank
            if interactive:
                next_rank = self.get_next_rank(weltanschauung, nummer)
                target_rank = self.ask_user_for_rank_priority(weltanschauung, nummer, next_rank)
                
                if target_rank == -1:  # User chose to skip
                    print(f"   ⏭️  Skipped {weltanschauung} #{nummer}")
                    return True
            else:
                target_rank = self.get_next_rank(weltanschauung, nummer)
            
            # Prepare document
            document = self.prepare_document(raw_data, target_rank)
            
            # Validate document
            is_valid, errors = self.validate_document(document)
            if not is_valid:
                print(f"   ❌ Validation failed for {weltanschauung} #{nummer}:")
                for error in errors:
                    print(f"      • {error}")
                return False
            
            # Adjust existing ranks if needed
            if target_rank < self.get_next_rank(weltanschauung, nummer):
                self.adjust_existing_ranks(weltanschauung, nummer, target_rank)
            
            # Insert document
            result = self.gedanken_collection.insert_one(document)
            
            if result.inserted_id:
                print(f"   ✅ Added {weltanschauung} #{nummer} with rank {target_rank}")
                return True
            else:
                print(f"   ❌ Failed to insert {weltanschauung} #{nummer}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error adding entry: {e}")
            return False
    
    def add_batch(self, data_list: List[Dict], interactive: bool = True) -> Dict[str, int]:
        """Add multiple entries in batch."""
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        print(f"\n🚀 Starting batch processing of {len(data_list)} entries...")
        
        for i, raw_data in enumerate(data_list, 1):
            print(f"\n[{i}/{len(data_list)}]", end=" ")
            
            if self.add_single_entry(raw_data, interactive):
                results['success'] += 1
            else:
                results['failed'] += 1
        
        print(f"\n📊 BATCH PROCESSING COMPLETE:")
        print(f"   ✅ Successful: {results['success']}")
        print(f"   ❌ Failed: {results['failed']}")
        print(f"   ⏭️  Skipped: {results['skipped']}")
        
        return results
    
    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            print("🔌 Database connection closed")

def load_gedankenfehler_umkehren_data(file_path: str) -> List[Dict]:
    """Load gedankenfehler-umkehren data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'results' in data:
            return data['results']
        else:
            print(f"❌ Unexpected data format in {file_path}")
            return []
            
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {file_path}: {e}")
        return []

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add gedankenfehler-umkehren results to database')
    parser.add_argument('data_file', help='JSON file containing gedankenfehler-umkehren results')
    parser.add_argument('--non-interactive', action='store_true', 
                       help='Run without user interaction (auto-assign ranks)')
    parser.add_argument('--mongodb-uri', help='MongoDB connection URI')
    
    args = parser.parse_args()
    
    # Load data
    data_list = load_gedankenfehler_umkehren_data(args.data_file)
    if not data_list:
        print("❌ No data to process")
        return 1
    
    # Initialize integrator
    integrator = GedankenfehlerUmkehrenIntegrator(args.mongodb_uri)
    if not integrator.connect():
        return 1
    
    try:
        # Process data
        results = integrator.add_batch(data_list, interactive=not args.non_interactive)
        
        # Success if at least some entries were processed successfully
        return 0 if results['success'] > 0 else 1
        
    finally:
        integrator.close()

if __name__ == "__main__":
    sys.exit(main()) 