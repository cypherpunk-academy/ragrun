# 🔄 Gedankenfehler-Umkehren Enhancement Plan

**Transformation from 3-Stage to 4-Stage AI Processing Pipeline**

---

## 📊 Current vs. New Architecture

### 🔄 **Current 3-Stage Pipeline:**

```
1. Reformulate → 3 variations from worldview perspective
2. Resolve → gedanke + gedanke_kurz + gedanke_einfach
3. Modernize → modern gedanke + modern gedanke_kurz
```

### ✨ **New 4-Stage Pipeline:**

```
1. Reformulate → 3 variations (unchanged)
2. Resolve → gedanke only (no kurz/einfach)
3. Parallel Processing:
   ├── Modernize → modern gedanke only (no kurz)
   ├── Simplify → gedanke_einfach (based on resolve)
   └── Glossary → glossar terms (based on resolve)
4. Summarize → gedanke_kurz (based on modernize)
```

---

## 🎯 Implementation Plan

### **Phase 1: Template Updates** ⚙️

#### **1.1 Update Resolve Template**

**File:** `assistants/templates/gedankenfehler-formulieren.mdt`

**Current Output:**

```json
{
    "gedanke": "300-word correction",
    "gedanke_zusammenfassung": "30-35 word summary",
    "gedanke_kind": "child-friendly explanation"
}
```

**New Output:**

```json
{
    "gedanke": "300-word correction"
}
```

**Template Changes:**

-   Remove `gedanke_zusammenfassung` and `gedanke_kind` from JSON format
-   Remove instructions about summary and child-friendly versions
-   Focus solely on the main philosophical correction

#### **1.2 Update Modernization Template**

**Current Output:**

```json
{
    "gedanke": "modernized version",
    "gedanke_kurz": "modern summary"
}
```

**New Output:**

```json
{
    "gedanke": "modernized version"
}
```

#### **1.3 Create New Templates**

**A. Simplification Template:** `gedankenfehler-einfach.mdt`

```jinja2
Bitte antworte ausschließlich mit einem **gültigen und kommentarlosen JSON-Objekt** im folgenden Format:

{
    "gedanke_einfach": "Kinderfreundliche Erklärung für 10-Jährige"
}

Erkläre folgenden philosophischen Text so, dass ihn ein 10-jähriges Kind verstehen kann:

** {{ gedanke }} **

Verwende einfache Worte, kurze Sätze und vertraute Beispiele. Behalte die philosophische Essenz bei, aber mache sie zugänglich. Verwende KEINE Metabezüge wie "Der Text zeigt" oder "Das bedeutet".
```

**B. Summary Template:** `gedankenfehler-kurz.mdt`

```jinja2
Bitte antworte ausschließlich mit einem **gültigen und kommentarlosen JSON-Objekt** im folgenden Format:

{
    "gedanke_kurz": "Kurze Zusammenfassung in 30-35 Worten"
}

Fasse folgenden philosophischen Text in 30-35 Worten zusammen:

** {{ gedanke }} **

Behalte die philosophische Präzision bei und erfasse die Kernaussage. Verwende KEINE Metabezüge wie "Der Text behandelt" oder "Es geht um".
```

### **Phase 2: Enhanced GedankenfehlerUmkehrenCommand Class** 🔧

#### **2.1 Core Class Restructure**

```python
class GedankenfehlerUmkehrenCommand:
    """Enhanced 4-stage gedankenfehler-umkehren processing"""

    def __init__(self):
        self.mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/12_weltanschauungen')
        self.client = None
        self.db = None
        self.assistant_manager = None

        # Processing results storage
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

    def setup_assistant_manager(self):
        """Initialize DeepSeek Assistant Manager"""
        from assistants.deepseek_assistant_manager import DeepSeekAssistantManager
        self.assistant_manager = DeepSeekAssistantManager()
```

#### **2.2 Stage Processing Methods**

```python
def stage_1_reformulate(self, gedanke: str, weltanschauung: str, assistant_id: str) -> dict:
    """Stage 1: Reformulate gedanke (unchanged from current)"""

def stage_2_resolve(self, chosen_reformulation: str, weltanschauung: str, assistant_id: str, aspekt: str = None) -> dict:
    """Stage 2: Resolve gedankenfehler (simplified output)"""

def stage_3_parallel_processing(self, resolve_result: dict, weltanschauung: str, assistant_id: str) -> dict:
    """Stage 3: Parallel processing of modernize, simplify, glossary"""

def stage_4_summarize(self, modernize_result: dict, weltanschauung: str, assistant_id: str) -> dict:
    """Stage 4: Create summary based on modernized version"""
```

### **Phase 3: Parallel Processing Implementation** ⚡

#### **3.1 Concurrent Execution**

```python
import asyncio
import concurrent.futures
from typing import Tuple

async def stage_3_parallel_processing(self, resolve_result: dict, weltanschauung: str, assistant_id: str) -> Tuple[dict, dict, dict]:
    """Execute modernize, simplify, and glossary in parallel"""

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        modernize_future = executor.submit(
            self._process_modernize,
            resolve_result['gedanke'], weltanschauung, assistant_id
        )

        simplify_future = executor.submit(
            self._process_simplify,
            resolve_result['gedanke'], weltanschauung, assistant_id
        )

        glossary_future = executor.submit(
            self._process_glossary,
            resolve_result['gedanke'], weltanschauung, assistant_id
        )

        # Collect results
        modernize_result = modernize_future.result()
        simplify_result = simplify_future.result()
        glossary_result = glossary_future.result()

        return modernize_result, simplify_result, glossary_result
```

#### **3.2 Individual Processing Methods**

```python
def _process_modernize(self, gedanke: str, weltanschauung: str, assistant_id: str) -> dict:
    """Process modernization without gedanke_kurz"""
    prompt = f"""Reformuliere folgenden authentischen philosophischen Text in moderne, zugängliche Sprache:

** {gedanke} **

JSON Format:
{{
    "gedanke": "Moderne Reformulierung ohne Metabezüge"
}}
"""
    return self._query_assistant(assistant_id, prompt, "modernize")

def _process_simplify(self, gedanke: str, weltanschauung: str, assistant_id: str) -> dict:
    """Process simplification for children"""
    from assistants.template_processor import TemplateProcessor
    processor = TemplateProcessor()

    prompt = processor.render_template(
        "gedankenfehler-einfach",
        weltanschauung,
        {"gedanke": gedanke}
    )
    return self._query_assistant(assistant_id, prompt, "simplify")

def _process_glossary(self, gedanke: str, weltanschauung: str, assistant_id: str) -> dict:
    """Process glossary extraction"""
    from assistants.template_processor import TemplateProcessor
    processor = TemplateProcessor()

    prompt = processor.render_template(
        "gedankenfehler-glossar",
        weltanschauung,
        {"korrektur": gedanke}
    )
    return self._query_assistant(assistant_id, prompt, "glossary")
```

### **Phase 4: Database Schema Updates** 🗄️

#### **4.1 Enhanced gedanken Collection**

```javascript
// Updated gedanken document structure
{
  "_id": ObjectId,
  "autor": String,
  "autorId": String,
  "weltanschauung": String,
  "created_at": Date,
  "ausgangsgedanke": String,
  "ausgangsgedanke_in_weltanschauung": String,
  "id": String,

  // Main content (Stage 2 + 3 + 4)
  "gedanke": String,                    // From resolve (authentic)
  "gedanke_modern": String,             // From modernize
  "gedanke_einfach": String,            // From simplify
  "gedanke_kurz": String,               // From summarize

  // Metadata
  "nummer": Number,
  "rank": Number,
  "stichwort": String,
  "aspekt": String,
  "model": String,

  // Processing information
  "processing_stages": {
    "reformulate": { "time": Number, "cost": Number },
    "resolve": { "time": Number, "cost": Number },
    "modernize": { "time": Number, "cost": Number },
    "simplify": { "time": Number, "cost": Number },
    "summarize": { "time": Number, "cost": Number }
  },
  "total_processing_time": Number,
  "total_cost": Number,
  "rag_citations": Array,

  // Glossary reference
  "glossary_terms": Array              // References to glossar collection
}
```

#### **4.2 New glossar Collection**

```javascript
// New glossar collection structure
{
  "_id": ObjectId,
  "begriff": String,                   // Term/concept
  "beschreibung": String,              // 50-80 word description
  "weltanschauung": String,            // Philosophical worldview
  "created_at": Date,
  "gedanken_id": String,               // Reference to gedanken document
  "gedanken_nummer": Number,           // Gedankenfehler number
  "autor": String,                     // Author who created this
  "id": String                         // Unique term ID
}
```

### **Phase 5: Enhanced Workflow Implementation** 🔄

#### **5.1 Main Processing Method**

```python
def process_gedankenfehler_umkehren(self, gedankenfehler: str, weltanschauung: str, nummer: int = None, aspekt: str = None) -> bool:
    """Complete 4-stage processing workflow"""

    try:
        # Setup
        self.setup_assistant_manager()
        assistant_id = self._get_assistant_id(weltanschauung)

        print("🚀 Starting 4-stage gedankenfehler-umkehren processing...")

        # Stage 1: Reformulate
        print("\n🔄 Stage 1: Reformulating gedanke...")
        start_time = time.time()
        reformulate_result = self.stage_1_reformulate(gedankenfehler, weltanschauung, assistant_id)
        self.timings['reformulate'] = time.time() - start_time
        self.results['reformulate'] = reformulate_result

        # User selection of reformulation
        chosen_reformulation = self._get_user_choice(reformulate_result['gedanken_in_weltanschauung'])

        # Stage 2: Resolve
        print("\n🔄 Stage 2: Resolving gedankenfehler...")
        start_time = time.time()
        resolve_result = self.stage_2_resolve(chosen_reformulation, weltanschauung, assistant_id, aspekt)
        self.timings['resolve'] = time.time() - start_time
        self.results['resolve'] = resolve_result

        # Stage 3: Parallel Processing
        print("\n🔄 Stage 3: Parallel processing (modernize, simplify, glossary)...")
        start_time = time.time()
        modernize_result, simplify_result, glossary_result = self.stage_3_parallel_processing(
            resolve_result, weltanschauung, assistant_id
        )
        parallel_time = time.time() - start_time

        self.results['modernize'] = modernize_result
        self.results['simplify'] = simplify_result
        self.results['glossary'] = glossary_result
        self.timings['parallel'] = parallel_time

        # Stage 4: Summarize
        print("\n🔄 Stage 4: Creating summary...")
        start_time = time.time()
        summarize_result = self.stage_4_summarize(modernize_result, weltanschauung, assistant_id)
        self.timings['summarize'] = time.time() - start_time
        self.results['summarize'] = summarize_result

        # Display results
        self._display_results()

        # Save to database
        return self._save_to_database(gedankenfehler, chosen_reformulation, weltanschauung, nummer, aspekt)

    except Exception as e:
        print(f"❌ Error in processing: {e}")
        return False
```

#### **5.2 Database Operations**

```python
def _save_to_database(self, original_gedanke: str, chosen_reformulation: str, weltanschauung: str, nummer: int, aspekt: str) -> bool:
    """Save results to both gedanken and glossar collections"""

    try:
        if not self.connect():
            return False

        # Prepare gedanken document
        gedanken_doc = {
            "autor": self.authors[weltanschauung],
            "autorId": self._get_author_id(weltanschauung),
            "weltanschauung": weltanschauung,
            "created_at": datetime.now(),
            "ausgangsgedanke": original_gedanke,
            "ausgangsgedanke_in_weltanschauung": chosen_reformulation,
            "id": str(uuid.uuid4()),

            # Main content
            "gedanke": self.results['resolve']['gedanke'],
            "gedanke_modern": self.results['modernize']['gedanke'],
            "gedanke_einfach": self.results['simplify']['gedanke_einfach'],
            "gedanke_kurz": self.results['summarize']['gedanke_kurz'],

            # Metadata
            "nummer": nummer or self.get_next_nummer(),
            "rank": self.get_next_rank(weltanschauung, nummer),
            "model": "gedankenfehler-umkehren-v2",
            "processing_stages": self.timings,
            "total_processing_time": sum(self.timings.values()),
            "total_cost": sum(self.costs.values())
        }

        # Insert gedanken document
        gedanken_result = self.db.gedanken.insert_one(gedanken_doc)
        gedanken_id = str(gedanken_result.inserted_id)

        # Save glossary terms
        glossary_terms = self.results['glossary'].get('glossar', [])
        glossary_ids = []

        for term in glossary_terms:
            glossar_doc = {
                "begriff": term['begriff'],
                "beschreibung": term['beschreibung'],
                "weltanschauung": weltanschauung,
                "created_at": datetime.now(),
                "gedanken_id": gedanken_doc['id'],
                "gedanken_nummer": gedanken_doc['nummer'],
                "autor": gedanken_doc['autor'],
                "id": str(uuid.uuid4())
            }

            glossar_result = self.db.glossar.insert_one(glossar_doc)
            glossary_ids.append(str(glossar_result.inserted_id))

        # Update gedanken document with glossary references
        self.db.gedanken.update_one(
            {"_id": gedanken_result.inserted_id},
            {"$set": {"glossary_terms": glossary_ids}}
        )

        print(f"\n✅ Successfully saved to database:")
        print(f"   🏷️  Gedanken ID: {gedanken_doc['id']}")
        print(f"   📚 Glossary terms: {len(glossary_terms)}")
        print(f"   🌍 Weltanschauung: {weltanschauung}")

        return True

    except Exception as e:
        print(f"❌ Database save error: {e}")
        return False
    finally:
        if self.client:
            self.client.close()
```

### **Phase 6: CLI Integration Updates** 💻

#### **6.1 Update rag-cli gedankenfehler-umkehren Command**

```python
@assistants_group.command('gedankenfehler-umkehren')
@click.option('--weltanschauung', '-w', required=True, help='Weltanschauung zur Auswahl des Assistenten')
@click.option('--output-format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.option('--output-file', help='Save output to file')
@click.option('--aspekt', help='Zusätzlicher Aspekt zur Berücksichtigung')
def assistants_gedankenfehler_umkehren_v2(weltanschauung: str, output_format: str, output_file: Optional[str], aspekt: Optional[str]):
    """Enhanced 4-stage gedankenfehler-umkehren processing"""

    try:
        # Import enhanced processor
        sys.path.insert(0, PROJECT_ROOT)
        from scripts.gedankenfehler_umkehren import GedankenfehlerUmkehrenCommand

        processor = GedankenfehlerUmkehrenCommand()

        # [Gedankenfehler selection logic remains the same]

        # Process using new 4-stage pipeline
        success = processor.process_gedankenfehler_umkehren(
            gedanke, weltanschauung, nummer=selected_fehler['nummer'], aspekt=aspekt
        )

        if success:
            click.echo("✅ 4-stage processing completed successfully!")
        else:
            click.echo("❌ Processing failed")

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
```

---

## 📋 Implementation Checklist

### **Phase 1: Template Updates**

-   [ ] Update `gedankenfehler-formulieren.mdt` (remove kurz/einfach)
-   [ ] Update modernization template (remove gedanke_kurz)
-   [ ] Create `gedankenfehler-einfach.mdt`
-   [ ] Create `gedankenfehler-kurz.mdt`
-   [ ] Test all templates with TemplateProcessor

### **Phase 2: Core Class Enhancement**

-   [ ] Restructure `GedankenfehlerUmkehrenCommand` class
-   [ ] Implement stage processing methods
-   [ ] Add performance tracking
-   [ ] Add result storage management

### **Phase 3: Parallel Processing**

-   [ ] Implement concurrent execution for Stage 3
-   [ ] Add individual processing methods
-   [ ] Test parallel performance vs sequential
-   [ ] Add error handling for parallel failures

### **Phase 4: Database Updates**

-   [ ] Update gedanken collection schema
-   [ ] Create glossar collection
-   [ ] Update database save operations
-   [ ] Add migration script for existing data

### **Phase 5: Workflow Integration**

-   [ ] Implement complete 4-stage workflow
-   [ ] Add user interaction for reformulation selection
-   [ ] Add result display and validation
-   [ ] Test end-to-end processing

### **Phase 6: CLI Updates**

-   [ ] Update rag-cli command
-   [ ] Add new command line options
-   [ ] Test CLI integration
-   [ ] Update documentation

---

## 🎯 Success Metrics

### **Performance Targets**

-   [ ] **Stage 3 Parallel Processing:** 60-70% time reduction vs sequential
-   [ ] **Total Processing Time:** < 15 seconds for complete workflow
-   [ ] **Database Operations:** < 2 seconds for save operations
-   [ ] **Memory Usage:** < 100MB increase during parallel processing

### **Quality Targets**

-   [ ] **Resolve Output:** Pure philosophical correction without summaries
-   [ ] **Modernization:** Contemporary language while preserving essence
-   [ ] **Simplification:** 10-year-old comprehension level
-   [ ] **Glossary:** 5 relevant terms with 50-80 word descriptions
-   [ ] **Summary:** Precise 30-35 word summaries

### **Data Integrity**

-   [ ] **Database Consistency:** All references between collections valid
-   [ ] **Rank Management:** Proper ranking system maintained
-   [ ] **Author Mapping:** Correct author assignments
-   [ ] **Processing Metadata:** Complete timing and cost tracking

---

## 🚀 Benefits of New Architecture

### **🎯 Enhanced Processing Quality**

-   **Specialized Focus:** Each stage optimized for specific output
-   **Better Results:** No conflicting requirements in single prompts
-   **Cleaner Separation:** Authentic vs. modern versions clearly separated

### **⚡ Performance Improvements**

-   **Parallel Execution:** 60-70% faster Stage 3 processing
-   **Efficient Resource Usage:** Better utilization of AI models
-   **Reduced Latency:** Concurrent processing reduces total time

### **📊 Better Data Organization**

-   **Separate Collections:** Glossary terms properly organized
-   **Rich Metadata:** Complete processing information tracked
-   **Clear References:** Proper relationships between documents

### **🔧 Improved Maintainability**

-   **Modular Design:** Each stage independently testable
-   **Clear Responsibilities:** Each method has single purpose
-   **Better Error Handling:** Isolated failure points

---

## ⚠️ Migration Considerations

### **Backward Compatibility**

-   Maintain old CLI command as `gedankenfehler-umkehren-legacy`
-   Provide data migration scripts for existing entries
-   Keep old database fields for transition period

### **Testing Strategy**

-   Unit tests for each processing stage
-   Integration tests for complete workflow
-   Performance benchmarks for parallel processing
-   Database integrity tests

### **Rollback Plan**

-   Keep current system operational during migration
-   Implement feature flags for gradual rollout
-   Database backup procedures before major changes

---

**🎊 This enhancement transforms the gedankenfehler-umkehren system into a sophisticated, parallel-processing pipeline that delivers higher quality results in less time while maintaining data integrity and extensibility.**
