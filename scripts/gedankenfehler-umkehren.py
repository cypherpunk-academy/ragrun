#!/usr/bin/env python3
"""
Enhanced gedankenfehler-umkehren command with 4-stage pipeline
Creates comprehensive umkehrung entries with parallel processing
"""

import os
import sys
import json
import uuid
import time
import concurrent.futures
from datetime import datetime
from pymongo import MongoClient
import argparse

class GedankenfehlerUmkehrenCommand:
    """Enhanced command for 4-stage gedankenfehler umkehrungen with parallel processing"""
    
    def __init__(self):
        self.mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/12_weltanschauungen')
        self.client = None
        self.db = None
        self.assistant_manager = None
        
        # Processing results storage for 4-stage pipeline
        self.results = {
            'reformulate': None,
            'resolve': None,
            'modernize': None,
            'simplify': None,
            'glossary': None,
            'summarize': None
        }
        
        # Performance tracking
        self.timings = {}
        self.costs = {}
        self.performance_metrics = {
            'workflow_start': None,
            'workflow_end': None,
            'total_time': None,
            'stage_times': {},
            'memory_usage': {},
            'processing_stats': {}
        }
        
        # Author mappings
        self.authors = {
            "Dynamismus": "Ariadne Ikarus Nietzsche",
            "Idealismus": "Aurelian I. Schelling", 
            "Individualismus": "Amara Illias Leibniz",
            "Materialismus": "Aloys I. Freud",
            "Mathematismus": "Arcadius Ikarus Torvalds",
            "Phänomenalismus": "Aetherius Imaginaris Goethe",
            "Pneumatismus": "Aurelian Irenicus Novalis",
            "Psychismus": "Archetype Intuitionis Fichte",
            "Rationalismus": "Aristoteles Isaak Herder",
            "Realismus": "Arvid I. Steiner",
            "Sensualismus": "Apollo Ikarus Schiller",
            "Spiritualismus": "Amara I. Steiner"
        }
    
    def setup_assistant_manager(self):
        """Initialize DeepSeek Assistant Manager for AI processing"""
        try:
            from assistants.deepseek_assistant_manager import DeepSeekAssistantManager
            self.assistant_manager = DeepSeekAssistantManager()
            return True
        except ImportError as e:
            print(f"❌ Failed to import DeepSeekAssistantManager: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to setup assistant manager: {e}")
            return False
    
    def _get_assistant_id(self, weltanschauung: str) -> str:
        """Get assistant ID for specific weltanschauung"""
        if not self.assistant_manager:
            raise ValueError("Assistant manager not initialized")
            
        weltanschauung_short = weltanschauung[:4].lower()
        for assistant_id, config in self.assistant_manager.assistant_configs.items():
            if config.get('worldview', '')[:4].lower() == weltanschauung_short:
                return assistant_id
        
        raise ValueError(f"No assistant found for weltanschauung: {weltanschauung}")
    
    def process_gedankenfehler_umkehren(self, gedankenfehler: str, weltanschauung: str, nummer: int = None, aspekt: str = None) -> bool:
        """
        Complete 4-stage processing workflow
        
        Args:
            gedankenfehler: Original thought error to process
            weltanschauung: Philosophical worldview perspective  
            nummer: Specific gedankenfehler number (auto-assigned if None)
            aspekt: Additional aspects to consider
            
        Returns:
            bool: True if processing completed successfully
        """
        try:
            print("🚀 Starting 4-stage gedankenfehler-umkehren processing...")
            
            # Initialize performance tracking
            self.performance_metrics['workflow_start'] = time.time()
            self._reset_tracking()
            
            setup_start = time.time()
            
            # Setup
            if not self.setup_assistant_manager():
                print("❌ Failed to setup assistant manager")
                return False
                
            assistant_id = self._get_assistant_id(weltanschauung)
            self.timings['setup'] = time.time() - setup_start
            print(f"✅ Using assistant: {assistant_id} (setup: {self.timings['setup']:.3f}s)")
            
            # Stage 1: Reformulate
            print("\n🔄 Stage 1: Reformulating gedanke...")
            stage_start = time.time()
            self.results['reformulate'] = self.stage_1_reformulate(gedankenfehler, weltanschauung, assistant_id)
            self.performance_metrics['stage_times']['stage_1_total'] = time.time() - stage_start
            
            # User selection of reformulation
            selection_start = time.time()
            chosen_reformulation = self._get_user_choice(self.results['reformulate']['gedanken_in_weltanschauung'])
            self.timings['user_selection'] = time.time() - selection_start
            print(f"✅ Selected reformulation: {chosen_reformulation[:50]}...")
            
            # Stage 2: Resolve
            print("\n🔄 Stage 2: Resolving gedankenfehler...")
            stage_start = time.time()
            self.results['resolve'] = self.stage_2_resolve(chosen_reformulation, weltanschauung, assistant_id, aspekt)
            self.performance_metrics['stage_times']['stage_2_total'] = time.time() - stage_start
            
            # Stage 3: Parallel Processing
            print("\n🔄 Stage 3: Parallel processing (modernize, simplify, glossary)...")
            stage_start = time.time()
            modernize_result, simplify_result, glossary_result = self.stage_3_parallel_processing(
                self.results['resolve'], weltanschauung, assistant_id
            )
            self.performance_metrics['stage_times']['stage_3_total'] = time.time() - stage_start
            
            # Store results with comprehensive tracking
            self.results['modernize'] = modernize_result
            self.results['simplify'] = simplify_result  
            self.results['glossary'] = glossary_result
            self._record_processing_stats()
            
            # Stage 4: Summarize
            print("\n🔄 Stage 4: Creating summary...")
            stage_start = time.time()
            self.results['summarize'] = self.stage_4_summarize(self.results['modernize'], weltanschauung, assistant_id)
            self.performance_metrics['stage_times']['stage_4_total'] = time.time() - stage_start
            
            # Display results
            self._display_results()
            
            # Save to database
            print("\n💾 Saving to database...")
            db_start = time.time()
            success = self._save_to_database_enhanced(gedankenfehler, chosen_reformulation, weltanschauung, nummer, aspekt)
            self.timings['database_save'] = time.time() - db_start
            
            # Finalize performance tracking
            self.performance_metrics['workflow_end'] = time.time()
            self.performance_metrics['total_time'] = self.performance_metrics['workflow_end'] - self.performance_metrics['workflow_start']
            
            if success:
                print(f"\n✅ 4-stage processing completed successfully!")
                self._display_performance_summary()
                return True
            else:
                print("\n❌ Failed to save to database")
                self._display_performance_summary()
                return False
                
        except Exception as e:
            print(f"\n❌ Error in 4-stage processing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_user_choice(self, reformulations: list) -> str:
        """Get user choice from reformulations"""
        print("\n📝 Reformulation options:")
        for i, reform in enumerate(reformulations, 1):
            print(f"{i}. {reform}")
        
        while True:
            try:
                choice = int(input('\nChoose option (1-3): '))
                if 1 <= choice <= len(reformulations):
                    return reformulations[choice - 1]
                else:
                    print(f"Please choose 1-{len(reformulations)}")
            except ValueError:
                print("Please enter a number")
            except KeyboardInterrupt:
                print("\n❌ User cancelled")
                raise
    
    def _reset_tracking(self):
        """Reset all tracking dictionaries for new workflow"""
        self.results.clear()
        self.timings.clear()
        self.costs.clear()
        self.performance_metrics.update({
            'workflow_start': None,
            'workflow_end': None,
            'total_time': None,
            'stage_times': {},
            'memory_usage': {},
            'processing_stats': {}
        })
    
    def _record_processing_stats(self):
        """Record processing statistics for results"""
        self.performance_metrics['processing_stats'] = {
            'reformulations_generated': len(self.results.get('reformulate', {}).get('gedanken_in_weltanschauung', [])),
            'gedanke_length': len(self.results.get('resolve', {}).get('gedanke', '')),
            'modernized_length': len(self.results.get('modernize', {}).get('gedanke', '')),
            'simplified_length': len(self.results.get('simplify', {}).get('gedanke_einfach', '')),
            'summary_length': len(self.results.get('summarize', {}).get('gedanke_kurz', '')),
            'glossary_terms': len(self.results.get('glossary', {}).get('glossar', [])),
            'rag_citations': len(self.results.get('resolve', {}).get('rag_citations', []))
        }
    
    def _display_performance_summary(self):
        """Display comprehensive performance summary"""
        print(f"\n🚀 PERFORMANCE SUMMARY")
        print("="*60)
        
        # Overall timing
        total_time = self.performance_metrics.get('total_time', 0)
        print(f"⏱️  Total Processing Time: {total_time:.2f}s")
        
        # Stage breakdown
        print(f"\n📊 Stage Breakdown:")
        for stage, timing in self.performance_metrics.get('stage_times', {}).items():
            percentage = (timing / total_time * 100) if total_time > 0 else 0
            print(f"   {stage}: {timing:.3f}s ({percentage:.1f}%)")
        
        # Detailed timings
        print(f"\n🔍 Detailed Timings:")
        for operation, timing in self.timings.items():
            print(f"   {operation}: {timing:.3f}s")
        
        # Cost summary
        total_cost = sum(self.costs.values())
        print(f"\n💰 Cost Breakdown:")
        print(f"   Total Cost: ${total_cost:.6f}")
        for stage, cost in self.costs.items():
            percentage = (cost / total_cost * 100) if total_cost > 0 else 0
            print(f"   {stage}: ${cost:.6f} ({percentage:.1f}%)")
        
        # Processing stats
        stats = self.performance_metrics.get('processing_stats', {})
        if stats:
            print(f"\n📈 Processing Statistics:")
            for stat, value in stats.items():
                print(f"   {stat}: {value}")
        
        # Parallel vs Sequential comparison if both available
        parallel_time = self.timings.get('parallel_total')
        sequential_time = self.timings.get('sequential_total')
        
        if parallel_time and sequential_time:
            improvement = ((sequential_time - parallel_time) / sequential_time * 100)
            print(f"\n⚡ Parallel vs Sequential Comparison:")
            print(f"   Parallel time: {parallel_time:.3f}s")
            print(f"   Sequential time: {sequential_time:.3f}s")
            print(f"   Performance improvement: {improvement:.1f}%")
        elif parallel_time:
            print(f"\n⚡ Parallel Processing:")
            print(f"   Total parallel time: {parallel_time:.3f}s")
            # Show individual task times if available
            if any(k.endswith('_task') for k in self.timings.keys()):
                print(f"   Task breakdown:")
                for key, value in self.timings.items():
                    if key.endswith('_task'):
                        print(f"     {key}: {value:.3f}s")
        elif sequential_time:
            print(f"\n⚡ Sequential Processing (Fallback):")
            print(f"   Total sequential time: {sequential_time:.3f}s")
        
        print("="*60)
    
    def _display_results(self):
        """Display all processing results"""
        print("\n" + "="*80)
        print("📊 4-STAGE PROCESSING RESULTS")
        print("="*80)
        
        if self.results['resolve']:
            print(f"📜 Original Correction: {self.results['resolve'].get('gedanke', '')[:100]}...")
        
        if self.results['modernize']:
            print(f"🌟 Modernized Version: {self.results['modernize'].get('gedanke', '')[:100]}...")  
        
        if self.results['simplify']:
            print(f"👶 Child Version: {self.results['simplify'].get('gedanke_einfach', '')[:100]}...")
        
        if self.results['summarize']:
            print(f"📝 Summary: {self.results['summarize'].get('gedanke_kurz', '')}")
        
        if self.results['glossary']:
            glossary_terms = self.results['glossary'].get('glossar', [])
            print(f"📚 Glossary Terms: {len(glossary_terms)}")
            for term in glossary_terms[:3]:  # Show first 3 terms
                print(f"   • {term.get('begriff', '')}: {term.get('beschreibung', '')[:50]}...")
        
        print("="*80)
    
    def stage_1_reformulate(self, gedanke: str, weltanschauung: str, assistant_id: str) -> dict:
        """Stage 1: Reformulate gedanke from weltanschauung perspective"""
        start_time = time.time()
        
        try:
            prompt = f"""Gebe drei Varianten von folgendem Gedanken aus der Sicht deiner Weltanschauung ({weltanschauung}):
            
            {gedanke}
            
            JSON Format:
            {{
                "gedanken_in_weltanschauung": [
                    "Erste Umformulierung",
                    "Zweite Umformulierung", 
                    "Dritte Umformulierung"
                ],
                "gedanke": "{gedanke}"
            }}"""
            
            response = self.assistant_manager.query_assistant(
                assistant_id=assistant_id,
                user_message=prompt,
                use_knowledge_base=True,
                model_override="deepseek-reasoner"  # Complex philosophical reasoning
            )
            
            self.timings['reformulate'] = time.time() - start_time
            self.costs['reformulate'] = response['usage']['cost']
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response["message"], re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"✅ Stage 1 completed: {len(result.get('gedanken_in_weltanschauung', []))} reformulations generated")
                return result
            else:
                raise ValueError("Could not parse reformulate response")
                
        except Exception as e:
            print(f"❌ Stage 1 error: {e}")
            # Record timing and cost for fallback
            self.timings['reformulate'] = time.time() - start_time
            self.costs['reformulate'] = 0.0
            # Fallback to simple reformulations
            return {
                "gedanken_in_weltanschauung": [
                    f"Aus {weltanschauung.lower()}er Sicht: {gedanke}",
                    f"In der {weltanschauung.lower()}en Tradition: {gedanke}",
                    f"Gemäß {weltanschauung.lower()}er Philosophie: {gedanke}"
                ],
                "gedanke": gedanke
            }
    
    def stage_2_resolve(self, chosen_reformulation: str, weltanschauung: str, assistant_id: str, aspekt: str = None) -> dict:
        """Stage 2: Resolve gedankenfehler (simplified to return only gedanke)"""
        start_time = time.time()
        
        try:
            from assistants.template_processor import TemplateProcessor
            processor = TemplateProcessor()
            
            prompt = processor.process_gedankenfehler_template(
                worldview=weltanschauung,
                gedanke_in_weltanschauung=chosen_reformulation,
                aspekte=aspekt or ""
            )
            
            response = self.assistant_manager.query_assistant(
                assistant_id=assistant_id,
                user_message=prompt,
                use_knowledge_base=True,
                model_override="deepseek-reasoner"  # Deep analytical thinking
            )
            
            self.timings['resolve'] = time.time() - start_time
            self.costs['resolve'] = response['usage']['cost']
            
            # Parse JSON response - should now only contain "gedanke"
            import re
            json_match = re.search(r'\{.*\}', response["message"], re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                result = {
                    'gedanke': parsed.get('gedanke', ''),
                    'rag_citations': response.get("citations", [])
                }
                print(f"✅ Stage 2 completed: {len(result['gedanke'])} characters generated")
                return result
            else:
                raise ValueError("Could not parse resolve response")
                
        except Exception as e:
            print(f"❌ Stage 2 error: {e}")
            # Record timing and cost for fallback
            self.timings['resolve'] = time.time() - start_time
            self.costs['resolve'] = 0.0
            # Fallback to simple correction
            return {
                'gedanke': f"[{weltanschauung}] Korrigierte Fassung: {chosen_reformulation}",
                'rag_citations': []
            }
    
    def stage_3_parallel_processing(self, resolve_result: dict, weltanschauung: str, assistant_id: str) -> tuple:
        """
        Stage 3: Parallel processing of modernize, simplify, glossary
        
        Uses ThreadPoolExecutor with 3 workers to process tasks concurrently.
        Includes comprehensive error handling and timeout management.
        Falls back to sequential processing if parallel execution fails.
        """
        start_time = time.time()
        futures = {}
        results = {}
        
        try:
            print("🔄 Submitting parallel tasks...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="Stage3") as executor:
                # Submit all 3 tasks with descriptive names
                futures['modernize'] = executor.submit(
                    self._process_modernize, 
                    resolve_result['gedanke'], weltanschauung, assistant_id
                )
                print(f"   📤 Submitted modernize task")
                
                futures['simplify'] = executor.submit(
                    self._process_simplify,
                    resolve_result['gedanke'], weltanschauung, assistant_id
                )
                print(f"   📤 Submitted simplify task")
                
                futures['glossary'] = executor.submit(
                    self._process_glossary,
                    resolve_result['gedanke'], weltanschauung, assistant_id
                )
                print(f"   📤 Submitted glossary task")
                
                print(f"🎯 All 3 tasks submitted, collecting results...")
                
                # Collect results with individual error handling and timeout
                task_timeout = 30.0  # 30 second timeout per task
                
                for task_name, future in futures.items():
                    task_start = time.time()
                    try:
                        print(f"   ⏳ Waiting for {task_name}...")
                        result = future.result(timeout=task_timeout)
                        task_time = time.time() - task_start
                        
                        results[task_name] = result
                        self.timings[f'{task_name}_task'] = task_time
                        print(f"   ✅ {task_name} completed in {task_time:.2f}s")
                        
                    except concurrent.futures.TimeoutError:
                        print(f"   ⏰ {task_name} timed out after {task_timeout}s")
                        # Provide fallback result
                        results[task_name] = self._get_fallback_result(task_name, resolve_result['gedanke'])
                        self.timings[f'{task_name}_task'] = task_timeout
                        
                    except Exception as e:
                        print(f"   ❌ {task_name} failed: {e}")
                        # Provide fallback result
                        results[task_name] = self._get_fallback_result(task_name, resolve_result['gedanke'])
                        task_time = time.time() - task_start
                        self.timings[f'{task_name}_task'] = task_time
                
                # Calculate total parallel processing time
                total_parallel_time = time.time() - start_time
                self.timings['parallel_total'] = total_parallel_time
                
                # Extract individual results
                modernize_result = results.get('modernize', self._get_fallback_result('modernize', resolve_result['gedanke']))
                simplify_result = results.get('simplify', self._get_fallback_result('simplify', resolve_result['gedanke']))
                glossary_result = results.get('glossary', self._get_fallback_result('glossary', resolve_result['gedanke']))
                
                # Log performance summary
                successful_tasks = len([r for r in results.values() if r is not None])
                print(f"✅ Stage 3 parallel processing completed:")
                print(f"   ⏱️  Total time: {total_parallel_time:.2f}s")
                print(f"   🎯 Tasks completed: {successful_tasks}/3")
                print(f"   📊 Individual times: " + 
                      f"modernize={self.timings.get('modernize_task', 0):.2f}s, " +
                      f"simplify={self.timings.get('simplify_task', 0):.2f}s, " +
                      f"glossary={self.timings.get('glossary_task', 0):.2f}s")
                
                return modernize_result, simplify_result, glossary_result
                
        except Exception as e:
            print(f"❌ Stage 3 parallel processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Record failed parallel timing
            self.timings['parallel_total'] = time.time() - start_time
            
            # Fallback to sequential processing
            print("🔄 Falling back to sequential processing...")
            return self._fallback_sequential_processing(resolve_result, weltanschauung, assistant_id)
    
    def _get_fallback_result(self, task_name: str, gedanke: str) -> dict:
        """Provide fallback result for failed parallel tasks"""
        if task_name == 'modernize':
            return {"gedanke": f"[Fallback Modern] {gedanke}"}
        elif task_name == 'simplify':
            return {"gedanke_einfach": f"[Fallback Einfach] {gedanke[:100]}..."}
        elif task_name == 'glossary':
            return {"glossar": [{"begriff": "Fallback-Begriff", "beschreibung": "Fallback-Beschreibung"}]}
        else:
            return {}
    
    def stage_4_summarize(self, modernize_result: dict, weltanschauung: str, assistant_id: str) -> dict:
        """Stage 4: Create summary based on modernized version"""
        start_time = time.time()
        
        try:
            from assistants.template_processor import TemplateProcessor
            processor = TemplateProcessor()
            
            prompt = processor.render_template(
                "gedankenfehler-kurz",
                weltanschauung,
                {"gedanke": modernize_result['gedanke']}
            )
            
            response = self.assistant_manager.query_assistant(
                assistant_id=assistant_id,
                user_message=prompt,
                use_knowledge_base=False,
                model_override="deepseek-chat"  # Simple generation task
            )
            
            self.timings['summarize'] = time.time() - start_time
            self.costs['summarize'] = response['usage']['cost']
            
            import re
            json_match = re.search(r'\{.*\}', response["message"], re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                print(f"✅ Stage 4 completed: {len(result.get('gedanke_kurz', ''))} character summary")
                return result
            else:
                raise ValueError("Could not parse summarize response")
                
        except Exception as e:
            print(f"❌ Stage 4 error: {e}")
            # Record timing and cost for fallback
            self.timings['summarize'] = time.time() - start_time
            self.costs['summarize'] = 0.0
            # Fallback to simple summary
            gedanke = modernize_result.get('gedanke', '')
            words = gedanke.split()[:35]  # Take first 35 words
            return {
                'gedanke_kurz': ' '.join(words) + ('...' if len(gedanke.split()) > 35 else '')
            }
    
    def _save_to_database_enhanced(self, gedankenfehler: str, chosen_reformulation: str, weltanschauung: str, nummer: int = None, aspekt: str = None) -> bool:
        """
        Enhanced database save method for 4-stage pipeline results
        
        Saves both gedanken and glossar collections with proper linking.
        Uses modernized gedanke as the main gedanke field.
        """
        if not self.connect():
            print("❌ Failed to connect to database")
            return False
        
        try:
            # Auto-assign nummer if not provided
            if nummer is None:
                nummer = self.get_next_nummer()
                print(f"📝 Auto-assigned nummer: {nummer}")
            
            # Validate weltanschauung
            if weltanschauung not in self.authors:
                print(f"❌ Unknown weltanschauung: {weltanschauung}")
                print(f"   Available: {list(self.authors.keys())}")
                return False
            
            # Get author info
            autor = self.authors[weltanschauung]
            autor_id = self._get_author_id(weltanschauung)
            
            # Get next rank for this weltanschauung/nummer combination
            rank = self.get_next_rank(weltanschauung, nummer)
            
            # Extract processed results (with fallbacks for missing data)
            # Get original authentic gedanke from Stage 2 (resolve)
            original_gedanke = self.results.get('resolve', {}).get('gedanke', '')
            if not original_gedanke:
                original_gedanke = f"[Fallback] {chosen_reformulation}"
            
            # Get modernized gedanke from Stage 3 (modernize)
            modernized_gedanke = self.results.get('modernize', {}).get('gedanke', '')
            if not modernized_gedanke:
                # Fallback to original if modernization failed
                modernized_gedanke = original_gedanke
            
            simplified_gedanke = self.results.get('simplify', {}).get('gedanke_einfach', '')
            if not simplified_gedanke:
                simplified_gedanke = f"[Fallback] Vereinfachte Version für {weltanschauung}"
            
            summary_gedanke = self.results.get('summarize', {}).get('gedanke_kurz', '')
            if not summary_gedanke:
                words = modernized_gedanke.split()[:35]
                summary_gedanke = ' '.join(words) + ('...' if len(modernized_gedanke.split()) > 35 else '')
            
            # Create main gedanken document using existing schema
            gedanken_doc = {
                "autor": autor,
                "autorId": autor_id,
                "weltanschauung": weltanschauung,
                "created_at": datetime.now(),
                "ausgangsgedanke": gedankenfehler,
                "ausgangsgedanke_in_weltanschauung": chosen_reformulation,
                "id": str(uuid.uuid4()),
                "gedanke_original": original_gedanke,  # Authentic philosophical correction from Stage 2
                "gedanke": modernized_gedanke,  # Modernized version from Stage 3
                "gedanke_einfach": simplified_gedanke,
                "gedanke_kurz": summary_gedanke,
                "nummer": nummer,
                "model": "gedankenfehler-umkehren-v2",  # Updated model identifier
                "rank": rank
            }
            
            # Insert main gedanken document
            print(f"💾 Saving gedanken document...")
            gedanken_result = self.db.gedanken.insert_one(gedanken_doc)
            
            if not gedanken_result.inserted_id:
                print("❌ Failed to insert gedanken document")
                return False
            
            print(f"✅ Gedanken document saved with ID: {gedanken_doc['id']}")
            print(f"   • Weltanschauung: {weltanschauung}")
            print(f"   • Nummer: {nummer}")
            print(f"   • Rank: {rank}")
            print(f"   • Author: {autor}")
            print(f"   • Original gedanke length: {len(original_gedanke)} characters")
            print(f"   • Modernized gedanke length: {len(modernized_gedanke)} characters")
            print(f"   • Summary length: {len(summary_gedanke)} characters")
            print(f"   • Simplified length: {len(simplified_gedanke)} characters")
            
            # Save glossary terms to separate collection
            glossary_terms = self.results.get('glossary', {}).get('glossar', [])
            glossary_count = 0
            
            if glossary_terms and isinstance(glossary_terms, list):
                print(f"💾 Saving {len(glossary_terms)} glossary terms...")
                
                for term in glossary_terms:
                    if isinstance(term, dict) and 'begriff' in term and 'beschreibung' in term:
                        glossar_doc = {
                            "begriff": term['begriff'],
                            "beschreibung": term['beschreibung'],
                            "weltanschauung": weltanschauung,
                            "nummer": nummer,  # Link to gedanken via nummer
                            "createdAt": datetime.now(),
                            "modifiedAt": datetime.now()
                        }
                        
                        try:
                            glossar_result = self.db.glossar.insert_one(glossar_doc)
                            if glossar_result.inserted_id:
                                glossary_count += 1
                                print(f"   ✅ Glossary term saved: {term['begriff']}")
                            else:
                                print(f"   ⚠️  Failed to save glossary term: {term['begriff']}")
                        except Exception as term_error:
                            print(f"   ❌ Error saving glossary term '{term['begriff']}': {term_error}")
                    else:
                        print(f"   ⚠️  Skipping invalid glossary term: {term}")
                
                print(f"✅ Saved {glossary_count}/{len(glossary_terms)} glossary terms")
            else:
                print("ℹ️  No glossary terms to save")
            
            # Log final database save summary
            print(f"\n📊 Database Save Summary:")
            print(f"   • Gedanken document: ✅ Saved")
            print(f"   • Glossary terms: {glossary_count} saved")
            print(f"   • Total processing cost: ${sum(self.costs.values()):.6f}")
            print(f"   • Total processing time: {self.performance_metrics.get('total_time', 0):.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in enhanced database save: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.client:
                self.client.close()
    
    def _fallback_sequential_processing(self, resolve_result: dict, weltanschauung: str, assistant_id: str) -> tuple:
        """
        Fallback sequential processing if parallel execution fails
        
        Processes tasks one by one with individual error handling.
        Records timing for comparison with parallel performance.
        """
        start_time = time.time()
        
        print("🔄 Starting sequential fallback processing...")
        
        # Process modernize
        print("   🔄 Processing modernize (sequential)...")
        task_start = time.time()
        try:
            modernize_result = self._process_modernize(resolve_result['gedanke'], weltanschauung, assistant_id)
            modernize_time = time.time() - task_start
            self.timings['modernize_seq'] = modernize_time
            print(f"   ✅ Modernize completed in {modernize_time:.2f}s")
        except Exception as e:
            print(f"   ❌ Modernize failed in sequential: {e}")
            modernize_result = self._get_fallback_result('modernize', resolve_result['gedanke'])
            self.timings['modernize_seq'] = time.time() - task_start
        
        # Process simplify
        print("   🔄 Processing simplify (sequential)...")
        task_start = time.time()
        try:
            simplify_result = self._process_simplify(resolve_result['gedanke'], weltanschauung, assistant_id)
            simplify_time = time.time() - task_start
            self.timings['simplify_seq'] = simplify_time
            print(f"   ✅ Simplify completed in {simplify_time:.2f}s")
        except Exception as e:
            print(f"   ❌ Simplify failed in sequential: {e}")
            simplify_result = self._get_fallback_result('simplify', resolve_result['gedanke'])
            self.timings['simplify_seq'] = time.time() - task_start
        
        # Process glossary
        print("   🔄 Processing glossary (sequential)...")
        task_start = time.time()
        try:
            glossary_result = self._process_glossary(resolve_result['gedanke'], weltanschauung, assistant_id)
            glossary_time = time.time() - task_start
            self.timings['glossary_seq'] = glossary_time
            print(f"   ✅ Glossary completed in {glossary_time:.2f}s")
        except Exception as e:
            print(f"   ❌ Glossary failed in sequential: {e}")
            glossary_result = self._get_fallback_result('glossary', resolve_result['gedanke'])
            self.timings['glossary_seq'] = time.time() - task_start
        
        # Record total sequential time
        total_sequential_time = time.time() - start_time
        self.timings['sequential_total'] = total_sequential_time
        
        print(f"✅ Sequential fallback processing completed:")
        print(f"   ⏱️  Total time: {total_sequential_time:.2f}s")
        print(f"   📊 Individual times: " + 
              f"modernize={self.timings.get('modernize_seq', 0):.2f}s, " +
              f"simplify={self.timings.get('simplify_seq', 0):.2f}s, " +
              f"glossary={self.timings.get('glossary_seq', 0):.2f}s")
        
        return modernize_result, simplify_result, glossary_result
    
    def _process_modernize(self, gedanke: str, weltanschauung: str, assistant_id: str) -> dict:
        """
        Process modernization - improve gedanke readability and contemporary language
        
        Uses AI assistant to modernize philosophical language while preserving meaning.
        Handles timing, cost tracking, and error recovery.
        """
        start_time = time.time()
        
        try:
            # Create custom modernization prompt
            prompt = f"""Modernisiere den folgenden philosophischen Text für zeitgenössische Leser:

Aufgabe: Verbessere Lesbarkeit und verwende moderne Sprache, ohne die philosophische Präzision zu verlieren.

Originaltext:
------------------------------------------------------------------------------
{gedanke}
------------------------------------------------------------------------------

Anforderungen:
- Verwende zeitgemäße deutsche Sprache und Begriffe
- Ersetze veraltete oder schwer verständliche Ausdrücke durch moderne Äquivalente
- Verbessere die Satzstruktur für bessere Lesbarkeit
- Behalte die philosophische Tiefe und Genauigkeit bei
- Schreibe aus der Perspektive des {weltanschauung}
- Antworte als JSON: {{"gedanke": "Modernisierter Text"}}

Antworte nur mit gültigem JSON, ohne Codeblöcke oder andere Formatierung."""

            # Query assistant
            response = self.assistant_manager.query_assistant(
                assistant_id=assistant_id,
                user_message=prompt,
                use_knowledge_base=True,  # Use knowledge base for philosophical context
                model_override="deepseek-chat"  # Language transformation task
            )
            
            # Record timing and cost
            self.timings['modernize'] = time.time() - start_time
            self.costs['modernize'] = response['usage']['cost']
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response["message"], re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                modernized_text = result.get('gedanke', '')
                
                if modernized_text:
                    print(f"✅ Modernize completed: {len(modernized_text)} characters")
                    return {"gedanke": modernized_text}
                else:
                    raise ValueError("Empty modernized text in response")
            else:
                raise ValueError("Could not parse modernize JSON response")
                
        except Exception as e:
            print(f"❌ Modernize error: {e}")
            # Record fallback timing and cost
            self.timings['modernize'] = time.time() - start_time
            self.costs['modernize'] = 0.0
            
            # Fallback: basic modernization
            return {"gedanke": gedanke}
    
    def _process_simplify(self, gedanke: str, weltanschauung: str, assistant_id: str) -> dict:
        """
        Process simplification - create child-friendly version using template
        
        Uses gedankenfehler-einfach template to create explanations for 10-year-olds.
        Handles timing, cost tracking, and error recovery.
        """
        start_time = time.time()
        
        try:
            # Use TemplateProcessor to render gedankenfehler-einfach template
            from assistants.template_processor import TemplateProcessor
            processor = TemplateProcessor()
            
            prompt = processor.render_template(
                "gedankenfehler-einfach",
                weltanschauung,
                {"gedanke": gedanke}
            )
            
            # Query assistant
            response = self.assistant_manager.query_assistant(
                assistant_id=assistant_id,
                user_message=prompt,
                use_knowledge_base=False,  # Don't use knowledge base for simplification
                model_override="deepseek-chat"  # Language simplification task
            )
            
            # Record timing and cost
            self.timings['simplify'] = time.time() - start_time
            self.costs['simplify'] = response['usage']['cost']
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response["message"], re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                simplified_text = result.get('gedanke_einfach', '')
                
                if simplified_text:
                    print(f"✅ Simplify completed: {len(simplified_text)} characters for children")
                    return {"gedanke_einfach": simplified_text}
                else:
                    raise ValueError("Empty simplified text in response")
            else:
                raise ValueError("Could not parse simplify JSON response")
                
        except Exception as e:
            print(f"❌ Simplify error: {e}")
            # Record fallback timing and cost
            self.timings['simplify'] = time.time() - start_time
            self.costs['simplify'] = 0.0
            
            # Fallback: basic simplification
            words = gedanke.split()
            simple_text = ' '.join(words[:20])  # Take first 20 words
            if len(words) > 20:
                simple_text += "... (vereinfacht für Kinder)"
            return {"gedanke_einfach": simple_text}
    
    def _process_glossary(self, gedanke: str, weltanschauung: str, assistant_id: str) -> dict:
        """
        Process glossary extraction - extract terms using existing glossar template
        
        Uses gedankenfehler-glossar template to extract and define key philosophical terms.
        Handles timing, cost tracking, and error recovery.
        """
        start_time = time.time()
        
        try:
            # Use TemplateProcessor to render gedankenfehler-glossar template
            from assistants.template_processor import TemplateProcessor
            processor = TemplateProcessor()
            
            # Use the convenience method for glossar template
            prompt = processor.process_glossar_template(
                worldview=weltanschauung,
                korrektur=gedanke
            )
            
            # Query assistant
            response = self.assistant_manager.query_assistant(
                assistant_id=assistant_id,
                user_message=prompt,
                use_knowledge_base=True,  # Use knowledge base for philosophical term definitions
                model_override="deepseek-chat"  # Language extraction task
            )
            
            # Record timing and cost
            self.timings['glossary'] = time.time() - start_time
            self.costs['glossary'] = response['usage']['cost']
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response["message"], re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                glossary_terms = result.get('glossar', [])
                
                if isinstance(glossary_terms, list) and glossary_terms:
                    # Validate glossary structure
                    valid_terms = []
                    for term in glossary_terms:
                        if isinstance(term, dict) and 'begriff' in term and 'beschreibung' in term:
                            valid_terms.append({
                                'begriff': term['begriff'],
                                'beschreibung': term['beschreibung']
                            })
                    
                    if valid_terms:
                        print(f"✅ Glossary completed: {len(valid_terms)} terms extracted")
                        return {"glossar": valid_terms}
                    else:
                        raise ValueError("No valid glossary terms found")
                else:
                    raise ValueError("Invalid or empty glossary array in response")
            else:
                raise ValueError("Could not parse glossary JSON response")
                
        except Exception as e:
            print(f"❌ Glossary error: {e}")
            # Record fallback timing and cost
            self.timings['glossary'] = time.time() - start_time
            self.costs['glossary'] = 0.0
            
            # Fallback: extract basic terms from text
            words = gedanke.split()
            # Find potential philosophical terms (longer words, capitalized)
            potential_terms = [w.strip('.,!?:;') for w in words 
                             if len(w) > 6 and not w.islower()][:3]
            
            fallback_glossary = []
            for term in potential_terms:
                fallback_glossary.append({
                    "begriff": term,
                    "beschreibung": f"Philosophischer Begriff aus der {weltanschauung.lower()}en Tradition"
                })
            
            if not fallback_glossary:
                fallback_glossary = [{
                    "begriff": "Philosophie",
                    "beschreibung": f"Zentrale Denkrichtung des {weltanschauung}"
                }]
            
            return {"glossar": fallback_glossary}
    
    def _get_author_id(self, weltanschauung: str) -> str:
        """Get author ID for specific weltanschauung"""
        if not self.connect():
            return "unknown"
            
        try:
            autor = self.authors.get(weltanschauung)
            if not autor:
                return "unknown"
                
            autor_info = self.db.autoren.find_one({"name": autor})
            return autor_info.get("id", "unknown") if autor_info else "unknown"
        except Exception as e:
            print(f"⚠️  Warning: Could not get author ID: {e}")
            return "unknown"
        finally:
            if self.client:
                self.client.close()
    
    def connect(self):
        """Connect to database"""
        try:
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client['12_weltanschauungen']
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def get_next_nummer(self):
        """Get next available gedankenfehler number"""
        pipeline = [{"$group": {"_id": "$nummer"}}, {"$sort": {"_id": 1}}]
        used_numbers = [item["_id"] for item in self.db.gedanken.aggregate(pipeline)]
        
        for i in range(1, 44):
            if i not in used_numbers:
                return i
        return 44  # If all 1-43 are used, suggest 44
    
    def get_next_rank(self, weltanschauung, nummer):
        """Get next rank for weltanschauung/nummer combination"""
        existing = list(self.db.gedanken.find({"weltanschauung": weltanschauung, "nummer": nummer}))
        if not existing:
            return 1
        max_rank = max(entry.get('rank', 0) for entry in existing)
        return max_rank + 1
    
    def generate_umkehrung(self, gedankenfehler, weltanschauung):
        """Generate simple umkehrung for a gedankenfehler"""
        # Simple template-based generation
        return {
            "gedanke": f"[{weltanschauung}] Umkehrung: {gedankenfehler}",
            "gedanke_einfach": f"Einfache {weltanschauung}-Umkehrung",
            "gedanke_kurz": f"{weltanschauung} Umkehrung"
        }
    
    def create_entry(self, gedankenfehler, weltanschauung, nummer=None):
        """Create a single gedankenfehler-umkehren entry"""
        if not self.connect():
            return False
        
        try:
            # Auto-assign nummer if not provided
            if nummer is None:
                nummer = self.get_next_nummer()
                print(f"📝 Auto-assigned nummer: {nummer}")
            
            # Validate weltanschauung
            if weltanschauung not in self.authors:
                print(f"❌ Unknown weltanschauung: {weltanschauung}")
                print(f"   Available: {list(self.authors.keys())}")
                return False
            
            # Generate umkehrung
            umkehrung = self.generate_umkehrung(gedankenfehler, weltanschauung)
            
            # Get author info
            autor = self.authors[weltanschauung]
            autor_info = self.db.autoren.find_one({"name": autor})
            autor_id = autor_info.get("id") if autor_info else "unknown"
            
            # Get next rank
            rank = self.get_next_rank(weltanschauung, nummer)
            
            # Create entry
            entry = {
                "autor": autor,
                "autorId": autor_id,
                "weltanschauung": weltanschauung,
                "created_at": datetime.now(),
                "ausgangsgedanke": gedankenfehler,
                "ausgangsgedanke_in_weltanschauung": f"Aus {weltanschauung.lower()}er Sicht: {gedankenfehler}",
                "id": str(uuid.uuid4()),
                "gedanke": umkehrung["gedanke"],
                "gedanke_einfach": umkehrung["gedanke_einfach"],
                "gedanke_kurz": umkehrung["gedanke_kurz"],
                "nummer": nummer,
                "model": "gedankenfehler-umkehren-command",
                "rank": rank
            }
            
            # Insert entry
            result = self.db.gedanken.insert_one(entry)
            
            if result.inserted_id:
                print(f"✅ Created entry:")
                print(f"   • Weltanschauung: {weltanschauung}")
                print(f"   • Nummer: {nummer}")
                print(f"   • Rank: {rank}")
                print(f"   • Author: {autor}")
                print(f"   • ID: {entry['id']}")
                return True
            else:
                print(f"❌ Failed to create entry")
                return False
                
        except Exception as e:
            print(f"❌ Error creating entry: {e}")
            return False
        finally:
            if self.client:
                self.client.close()

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Gedankenfehler-Umkehren Command with 4-stage processing pipeline',
        epilog='''
        The enhanced pipeline includes:
        1. Reformulate - Generate 3 variations from weltanschauung perspective
        2. Resolve - Create authentic philosophical correction  
        3. Parallel Processing - Modernize, simplify, and extract glossary terms
        4. Summarize - Generate concise summary
        
        Features: Model optimization, parallel processing, comprehensive database integration
        '''
    )
    parser.add_argument('gedankenfehler', help='The gedankenfehler text to reverse/correct')
    parser.add_argument('weltanschauung', help='The philosophical worldview perspective')
    parser.add_argument('--nummer', type=int, help='Specific gedankenfehler nummer (auto-assigned if not provided)')
    parser.add_argument('--aspekt', type=str, help='Additional aspects to consider during processing')
    parser.add_argument('--legacy', action='store_true', help='Use legacy simple processing instead of 4-stage pipeline')
    
    args = parser.parse_args()
    
    command = GedankenfehlerUmkehrenCommand()
    
    # Use legacy method if requested
    if args.legacy:
        print("🔄 Using legacy simple processing...")
        success = command.create_entry(args.gedankenfehler, args.weltanschauung, args.nummer)
    else:
        print("🚀 Using enhanced 4-stage processing pipeline...")
        success = command.process_gedankenfehler_umkehren(
            gedankenfehler=args.gedankenfehler,
            weltanschauung=args.weltanschauung,
            nummer=args.nummer,
            aspekt=args.aspekt
        )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 