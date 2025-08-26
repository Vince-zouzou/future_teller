from langchain_community.chat_models.openai import ChatOpenAI
from typing import List, Dict, Any, Optional, Tuple
import json
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import BaseTool
import networkx as nx
import re
import streamlit as st

# Tavily search tool configuration
TAVILY_API_KEY = st.secrets["tavily_key"]
TAVILY_API_URL = "https://api.tavily.com/search"

class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = "Use Tavily search engine to search for related information"
    
    def _run(self, query: str) -> List[Dict[str, Any]]:
        """Execute Tavily search"""
        #st.write(query)
        headers = {
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "max_results": 10,
            "include_answer": False,
            "include_images": False,
            "topic": "news"
        }

        try:
            response = requests.post(TAVILY_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Format search results
            formatted_results = []
            for result in data.get('results', []):
                formatted_results.append({
                    "title": result.get('title', ''),
                    "url": result.get('url', ''),
                    "content": result.get('content', ''),
                    "published_date": result.get('published_date', ''),
                    "score": result.get('score', 0)
                })
            
            return formatted_results
            
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

class KnowledgeGraphTool(BaseTool):
    name: str = "knowledge_graph_query"
    description: str = "Query knowledge graph information"
    
    def _run(self, entities_str: str) -> Dict[str, Any]:
        """Query knowledge graph (using simple in-memory knowledge graph example)"""
        
        # Convert string to entity list
        try:
            entities = json.loads(entities_str)
        except:
            # 如果JSON解析失败，使用简单的分割
            entities = entities_str.split(',')
        
        # This can be replaced with actual knowledge graph query logic
        # Example: Return related relationships based on entity names
        knowledge_base = {
            "Trump": {
                "type": "Person",
                "attributes": ["Former US President", "Businessman", "Politician"],
                "relationships": {
                    "Tariff War": {"type": "Initiated", "weight": 0.9},
                    "Musk": {"type": "Knows", "weight": 0.7}
                }
            },
            "Musk": {
                "type": "Person",
                "attributes": ["Tesla CEO", "SpaceX Founder", "Entrepreneur"],
                "relationships": {
                    "Tesla": {"type": "Founded", "weight": 1.0},
                    "Tariff War": {"type": "Affected by", "weight": 0.8}
                }
            }
        }
        
        result = {}
        for entity in entities:
            if entity.strip() in knowledge_base:
                result[entity.strip()] = knowledge_base[entity.strip()]
        
        return result

class ProbabilityCalculatorTool(BaseTool):
    name: str = "probability_calculation"
    description: str = "计算事件发生的概率"
    
    def _run(self, input_data: str) -> Dict[str, float]:
        """基于上下文信息计算事件概率"""
        
        # 解析输入数据
        try:
            data = json.loads(input_data)
            event_description = data.get("event_description", "")
            context = data.get("context", {})
        except:
            # 如果解析失败，使用原始字符串作为事件描述
            event_description = input_data
            context = {}
        
        # 这里使用基于规则的概率估算
        # 实际应用中可以使用更复杂的模型
        
        probabilities = {}
        
        # Example probability calculation logic
        if "tariff war" in event_description and "Trump" in event_description:
            probabilities["High probability of occurrence"] = 0.7
            probabilities["Medium probability of occurrence"] = 0.2
            probabilities["Low probability of occurrence"] = 0.1

        elif "bankruptcy" in event_description and "Musk" in event_description:
            probabilities["High probability of occurrence"] = 0.3
            probabilities["Medium probability of occurrence"] = 0.4
            probabilities["Low probability of occurrence"] = 0.3

        else:
            # Default probability distribution
            probabilities["Occurrence"] = 0.5
            probabilities["Non-occurrence"] = 0.5
            
        return probabilities

class KnowledgeGraphConverter:
    """Knowledge Graph Converter - Uses AI to convert text information into knowledge graph triplets"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def text_to_triplets(self, text: str, query: str = "") -> List[Tuple[str, str, str]]:
        """Use AI to convert text to knowledge graph triplets"""
        prompt = f"""
        Based on the following text content and query context, extract knowledge graph triplets (subject-relation-object):
        
        Query context: {query}
        Text content: {text[:4000]}  # Limit text length
        
        Extract important entity relationship triplets, focusing on:
        1. Relationships between people, organizations, events
        2. Causal relationships, influence relationships, cooperation relationships
        3. Temporal sequences and logical relationships
        4. Specific behaviors and actions
        
        Return format must be strict Python list format:
        [("subject", "relation", "object"), ("subject", "relation", "object"), ...]
        
        Each triplet should contain:
        - Subject: specific entity or concept
        - Relation: clear relationship description (e.g., influences, causes, cooperates with, belongs to)
        - Object: specific entity or concept
        
        Example:
        [("Trump", "initiated", "tariff war"), ("tariff war", "affected", "global economy"), ("Musk", "was affected by", "tariff war")]
        
        Ensure the extracted relationships are accurate, specific, and meaningful in English.
        """
        
        try:
            response = self.llm.invoke(prompt)
            
            # 从响应中提取三元组
            import ast
            triplets = ast.literal_eval(response.content)
            
            if isinstance(triplets, list) and all(len(item) == 3 for item in triplets):
                return triplets
            else:
                # 如果格式不正确，返回空列表
                return []
                
        except Exception as e:
            print(f"AI知识图谱转换失败: {e}")
            # 失败时使用备用的规则匹配
            return self._fallback_text_to_triplets(text)
    
    def _fallback_text_to_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """备用的规则匹配方法"""
        triplets = []
        
        # 简单的规则匹配
        patterns = [
            # 人物-动作-对象模式
            (r'(\w+)(?:的|)(?:关税战|政策|决定)(?:对|影响)(\w+)', '影响', '对象'),
            (r'(\w+)(?:可能|将会)(导致|造成)(\w+)', '导致', '结果'),
            (r'(\w+)(?:和|与)(\w+)(?:的|)(关系|合作)', '关系', '对象'),
            (r'(\w+)(?:属于|是)(\w+)', '属于', '类别'),
            (r'(\w+)(?:在|于)(\w+)(?:发生)', '发生在', '地点'),
        ]
        
        for pattern, relation, _ in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    subject = match.group(1)
                    obj = match.group(2)
                    triplets.append((subject, relation, obj))
        
        return triplets
    
    @staticmethod
    def create_knowledge_graph(triplets: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """创建知识图谱网络"""
        G = nx.DiGraph()
        
        for subject, relation, obj in triplets:
            G.add_node(subject, type="entity")
            G.add_node(obj, type="entity")
            G.add_edge(subject, obj, relation=relation, label=relation)
        
        return G

class SupervisorAgent:
    """Supervisor Agent - Determines user intent and routes to appropriate workflow"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def analyze_intent(self, query: str) -> str:
        """Analyze user intent"""
        prompt = f"""
        Analyze the intent of the following user query and return the corresponding processing type:
        
        Available types:
        - workflow1: Asking about potential future impacts of events and the likelihood of events occurring  (e.g., What impact will Trump's tariff war have on the world landscape)
        - normal: Casual chat or simple Q&A (e.g., hello hows your day)
        
        User query: {query}
        
        Return only one of: workflow1 or normal.
        """
        
        response = self.llm.invoke(prompt)
        intent = response.content.strip().lower()
        
        if intent in ['workflow1', 'workflow2', 'normal']:
            return intent
        else:
            # 默认处理复杂查询
            if any(keyword in query.lower() for keyword in ['影响', '结果', '将会', '可能']):
                return 'workflow1'
            elif any(keyword in query.lower() for keyword in ['是否', '会不会', '可能性', '概率']):
                return 'workflow2'
            else:
                return 'normal'

class Workflow2Agent:
    """Workflow 1 - Analyzes potential future impacts of events"""
    
    def __init__(self, llm, prob_tool, kg_tool):
        self.llm = llm
        self.prob_tool = prob_tool
        self.kg_tool = kg_tool
    
    def process(self, query: str, status_callback=None) -> Dict[str, Any]:
        """处理workflow2查询"""
        # 提取关键实体
        if status_callback:
            status_callback("Extracting key entities from query")
        entities = self._extract_entities(query)
        
        # 查询知识图谱 - 确保传入正确的参数类型
        if status_callback:
            status_callback("Querying knowledge graph for entity relationships")
        entities_str = json.dumps(entities)
        kg_info = self.kg_tool.run(entities_str)
        
        # 计算概率 - 确保传入正确的参数类型
        if status_callback:
            status_callback("Calculating event probabilities based on context")
        prob_input = json.dumps({
            "event_description": query,
            "context": kg_info
        })
        probabilities = self.prob_tool.run(prob_input)
        
        # 生成分析报告
        if status_callback:
            status_callback("Generating comprehensive probability analysis report")
        report = self._generate_report(query, probabilities, kg_info)
        
        return {
            "probabilities": probabilities,
            "knowledge_graph": kg_info,
            "analysis_report": report
        }

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        prompt = f"""
        Extract key entity names from the following query:
        Query: {query}
        
        Return entity names in JSON array format, e.g.: ["entity1", "entity2", "entity3"]
        """
        
        response = self.llm.invoke(prompt)
        return json.loads(response.content)

    def _generate_report(self, query: str, probabilities: Dict[str, float], kg_info: Dict) -> str:
        """Generate probability analysis report"""
        prompt = f"""
        Based on the following information, generate a detailed analysis report:
        
        Original query: {query}
        Probability analysis results: {json.dumps(probabilities, ensure_ascii=False)}
        Knowledge graph information: {json.dumps(kg_info, ensure_ascii=False)}
        
        Please provide a detailed probability analysis report in the following format:

        ### Short-term (next 1-6 months)
        #### Step 1 & 2: Possible scenarios and probabilities
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing triplets) |
        |----------|------|--------------|------|-------------------|
        | 1 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 2 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 3 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 4 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 5 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 6 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 7 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 8 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |

        ### Medium-term (6 months - 2 years)
        #### Step 1 & 2: Possible scenarios and probabilities
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing triplets) |
        |----------|------|--------------|------|-------------------|
        | 1 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 2 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 3 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 4 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 5 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 6 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 7 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 8 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |

        ### Long-term (2+ years)
        #### Step 1 & 2: Possible scenarios and probabilities
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing triplets) |
        |----------|------|--------------|------|-------------------|
        | 1 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 2 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 3 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 4 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 5 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 6 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 7 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 8 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |

        Requirements:
        1. Provide 5-8 main scenarios for each time period (minimum 5, maximum 8)
        2. Total probability for each time period should be 100%
        3. Clearly cite knowledge graph triplets as basis
        4. Scenario descriptions should be specific, detailed, and have practical basis
        5. Respond in English
        """
        
        response = self.llm.invoke(prompt)
        return response.content

class Workflow1Agent:
    """Workflow 2 - Calculates the likelihood of event occurrence"""
    
    def __init__(self, llm, search_tool, kg_tool, prob_tool):
        self.llm = llm
        self.search_tool = search_tool
        self.kg_tool = kg_tool
        self.prob_tool = prob_tool
        self.converter = KnowledgeGraphConverter(llm)
    
    def process(self, query: str, status_callback=None) -> Dict[str, Any]:
        back = st.session_state.back
        """处理workflow1查询 - 按照时间线梳理 → 补充搜索 → 构建三元组 → 分析的流程"""
        # Step 1: 初步搜索建立时间线框架
        if status_callback:
            status_callback("🔍 Start initial research")
        initial_keywords = self._generate_timeline_search_terms(query)
        initial_results = []
        for keyword in initial_keywords:
            results = self.search_tool.run(keyword)
            initial_results.extend(results)
        back.write("Initial seach result:")
        back.write(initial_results)
        

        # Step 2: 创建初步时间线
        if status_callback:
            status_callback("📅 Create timeline")
        timeline = self._create_detailed_timeline(initial_results)
        back.write("timeline:")
        back.write(timeline)
            
        # Step 3: 基于时间线信息补充搜索
        if status_callback:
            status_callback("🔍 Find More information based on current information")
        supplemental_keywords = self._generate_supplemental_search_terms(query, timeline)
        supplemental_results = []
        for keyword in supplemental_keywords:
            results = self.search_tool.run(keyword)

            supplemental_results.extend(results)
        back.write("supplemental_results:")
        back.write(supplemental_results)


        # Step 4: 更新和完善时间线
        if status_callback:
            status_callback("📊 Update timeline and information")
        updated_timeline = self._update_timeline_with_supplemental(timeline, supplemental_results)
        back.write("updated_timeline:")
        back.write(updated_timeline)

        # Step 5: 从所有信息中提取三元组
        if status_callback:
            status_callback("🔗 Extract Relationship")
        all_results = initial_results + supplemental_results
        triplets = self._extract_triplets_from_all_sources(all_results, query)
        back.write("triplets:")
        back.write(triplets)

        # Step 6: 构建知识图谱
        if status_callback:
            status_callback("🕸️ Constract Knowledge graph")
        kg_data = self._create_knowledge_graph_from_triplets(triplets)
        back.write("kg_data:")
        back.write(kg_data)

        # 计算概率 - 确保传入正确的参数类型
        if status_callback:
            status_callback("Calculating event probabilities based on context")
        prob_input = json.dumps({
            "event_description": query,
            "context": kg_data
        })
        probabilities = self.prob_tool.run(prob_input)
        back.write("probabilities:")
        back.write(probabilities)
        
        # Step 7: 基于完整信息进行最终分析
        if status_callback:
            status_callback("📈 Final analysis")
        impact_analysis = self._analyze_impact_with_context(query, updated_timeline, kg_data, probabilities)
        back.write("impact_analysis:")
        back.write(impact_analysis)  

        return {
            "timeline": updated_timeline,
            "knowledge_graph": kg_data,
            "triplets": triplets,
            "impact_analysis": impact_analysis,
            "search_results": all_results
        }
    
    def _generate_timeline_search_terms(self, query: str) -> List[str]:
        """Generate search keywords for building timeline framework"""
        prompt = f"""
        You are a professional search query optimization expert. Please generate 3-5 most effective Tavily search keywords for the following query:

        Original query: {query}

        Optimization requirements:
        1. Translate original query to English and use it for searching
        2. Include specific time ranges or dates (e.g., "2024", "last 3 months")
        3. Use precise event names and key entities
        4. Include timeline-related keywords ("timeline", "chronology", "sequence of events")
        5. Use English keywords for better search results
        6. Avoid overly broad queries, maintain specificity

        Examples:
        - For "Apple company development history" → ["Apple Inc historical timeline 2020-2024", "Apple major events chronology", "Timeline of Apple product releases"]
        - For "COVID-19 pandemic development" → ["COVID-19 pandemic timeline 2020-2024", "Coronavirus key events chronology", "Major COVID-19 milestones timeline"]

        Return keywords in JSON array format in English.
        """
        
        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except:
            return [f"{query} timeline 2020-2024", f"{query} key events chronology", f"{query} major milestones"]

    def _generate_supplemental_search_terms(self, query: str, timeline: List[Dict]) -> List[str]:
        """基于时间线信息生成补充搜索关键词"""
        # 从时间线中提取关键实体和事件
        timeline_text = "\n".join([f"{item.get('date', '')}: {item.get('title', '')}" for item in timeline[:5]])
        
        prompt = f"""
        You are a professional search query optimization expert. Based on the following timeline information and original query, generate 3-5 most effective Tavily search keywords:

        Original query: {query}
        Preliminary timeline:
        {timeline_text}

        Optimization requirements:
        1. Translate original query to English and use it for searching
        2. Generate precise search queries for specific events and entities in the timeline
        3. Include keywords related to causal relationships, impact analysis, and detailed insights
        4. Use English keywords for better search results
        5. Include specific time ranges or event names
        6. Avoid duplicate timeline searches, focus on supplementary details

        Examples:
        - For "iPhone release" event in timeline → ["iPhone market impact analysis", "Apple iPhone sales statistics", "iPhone technological innovations"]
        - For "pandemic outbreak" event in timeline → ["COVID-19 economic impact studies", "Pandemic healthcare system effects", "Coronavirus vaccine efficacy data"]

        Return keywords in JSON array format in English.
        """
        
        response = self.llm.invoke(prompt)
        try:
            return json.loads(response.content)
        except:
            # 基于时间线内容生成更具体的后备关键词
            timeline_keywords = []
            for item in timeline[:3]:
                title = item.get('title', '')
                if title:
                    timeline_keywords.extend([f"{title} impact analysis", f"{title} detailed report", f"{title} statistics"])
            
            return timeline_keywords if timeline_keywords else [f"{query} impact analysis", f"{query} detailed report", f"{query} statistics"]

    def _create_detailed_timeline(self, search_results: List[Dict]) -> List[Dict]:
        """使用AI直接整理搜索结果成Timeline Table格式（Time, Description, SourceLink）"""
        try:

            prompt = f"""
            请分析以下搜索结果，整理成结构化的时间线表格(Timeline Table)。要求：

            输入数据：{json.dumps(search_results, ensure_ascii=False, indent=2)}
            
            输出要求：
            1. 从内容中提取或推断时间信息（Time），格式为YYYY-MM-DD
            2. 提取关键事件描述（Description），简洁明了
            3. 保留来源链接（SourceLink）
            4. 按时间顺序排序
            5. 如果无法确定具体日期，可以使用月份或年份
            6. 确保每个条目包含Time、Description、SourceLink三个字段
            
            请返回JSON格式的时间线数组，每个条目格式：
            {{
                "Time": "2024-01-15",
                "Description": "事件描述",
                "SourceLink": "https://example.com"
            }}
            
            只返回JSON数组，不要其他内容。
            """
            
            # 使用AI模型处理
            response = self.llm.invoke(prompt)
            
            # 解析AI返回的JSON
            timeline = json.loads(response.content.strip())
            
            
            return timeline
            
        except Exception as e:

            return self._create_fallback_timeline(search_results)
    
    def _create_fallback_timeline(self, search_results: List[Dict]) -> List[Dict]:
        """备用方法：原始的时间线创建逻辑"""
        timeline = []
        
        for result in search_results:
            if isinstance(result, dict):
                # 尝试从各种字段中提取时间信息
                time_field = None
                for field in ['published_date', 'date', 'timestamp', 'created_at']:
                    if field in result and result[field]:
                        time_field = result[field]
                        break
                
                if time_field:
                    try:
                        # 简化时间处理
                        time_str = str(time_field)[:10]  # 取前10个字符
                        timeline.append({
                            "Time": time_str,
                            "Description": result.get('title', result.get('content', '')[:100]),
                            "SourceLink": result.get('url', '')
                        })
                    except:
                        continue
        
        # 按时间排序
        timeline.sort(key=lambda x: x.get('Time', ''))
        return timeline

    def _update_timeline_with_supplemental(self, timeline: List[Dict[str, Any]], supplemental_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用AI结合已有时间线和补充信息来完善Timeline table"""
        if not supplemental_results:
            #st.write("⚠️ 补充搜索结果为空，无需更新时间线")
            return timeline
        
        #st.write(f"📊 开始使用AI完善时间线，已有 {len(timeline)} 条记录，新增 {len(supplemental_results)} 条补充结果")
        
        try:
            # 构建AI提示词，让AI结合已有时间线和补充信息来完善时间线
            prompt = f"""
            请结合以下已有的时间线数据和新的补充搜索结果，完善和更新时间线表格：

            已有时间线数据：
            {json.dumps(timeline, ensure_ascii=False, indent=2)}

            新的补充搜索结果：
            {json.dumps(supplemental_results, ensure_ascii=False, indent=2)}

            处理要求：
            1. 整合已有时间线和新的补充信息
            2. 去重相同时间点的条目
            3. 补充缺失的时间信息（从内容中推断）
            4. 完善事件描述的完整性和准确性
            5. 确保每个条目包含Time、Description、SourceLink三个字段
            6. 按时间顺序排序

            输出格式：JSON数组，每个条目格式：
            {{
                "Time": "2024-01-15",
                "Description": "完整的事件描述",
                "SourceLink": "https://example.com"
            }}

            请返回完善后的时间线JSON数组，不要其他内容。
            """
            
            # 使用AI模型处理
            response = self.llm.invoke(prompt)
            
            # 解析AI返回的JSON
            updated_timeline = json.loads(response.content.strip())
            

            return updated_timeline
            
        except Exception as e:

            return self._update_fallback_timeline(timeline, supplemental_results)
    
    def _update_fallback_timeline(self, timeline: List[Dict[str, Any]], supplemental_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """备用方法：原始的时间线更新逻辑"""
        new_timeline = timeline.copy()
        
        for result in supplemental_results:
            if isinstance(result, dict):
                # 尝试从各种字段中提取时间信息
                time_field = None
                for field in ['published_date', 'date', 'timestamp', 'created_at']:
                    if field in result and result[field]:
                        time_field = result[field]
                        break
                
                if time_field:
                    try:
                        time_str = str(time_field)[:10]
                        # 检查是否已有该时间的条目
                        existing_entry = next((item for item in new_timeline if item.get('Time') == time_str), None)
                        
                        if existing_entry:
                            # 更新现有条目描述
                            existing_desc = existing_entry.get('Description', '')
                            new_info = result.get('title', result.get('content', '')[:100])
                            if new_info not in existing_desc:
                                existing_entry['Description'] = f"{existing_desc} | {new_info}"
                        else:
                            # 添加新条目
                            new_timeline.append({
                                "Time": time_str,
                                "Description": result.get('title', result.get('content', '')[:100]),
                                "SourceLink": result.get('url', '')
                            })
                    except:
                        continue
        
        # 按时间排序
        new_timeline.sort(key=lambda x: x.get('Time', ''))
        return new_timeline

    def _generate_ai_structured_timeline(self, timeline: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """使用AI分析生成结构化的时间线表格"""
        if not timeline:
            return timeline
            
        prompt = f"""
        You are a professional historical event analyst. Based on the following timeline information and original query, generate a structured timeline table:

        Original query: {query}
        Search results timeline:
        {timeline}

        Please analyze this information and generate a clearer, more organized timeline with the following fields:
        1. Date (format: YYYY-MM-DD)
        2. Event title (concise and clear)
        3. Event description (detailed explanation)
        4. Source link (if available)
        5. Importance score (1-10 points)

        Return in JSON array format, each object containing: date, title, description, source, importance_score

        Note: Maintain event authenticity and accuracy, do not add fictional information.
        """
        
        try:
            response = self.llm.invoke(prompt)
            structured_timeline = json.loads(response.content)
            
            # 合并原始来源信息
            for item in structured_timeline:
                original_item = next((t for t in timeline if t['date'] == item.get('date', '')), None)
                if original_item and 'source' in original_item:
                    item['source'] = original_item['source']
            
            return structured_timeline
        except:
            # 如果AI分析失败，返回原始时间线
            return timeline

    def _extract_triplets_from_all_sources(self, search_results: List[Dict[str, Any]], query: str) -> List[Tuple[str, str, str]]:
        """从所有搜索结果中提取知识图谱三元组"""
        all_text = ' '.join([str(result.get('content', '')) for result in search_results if isinstance(result, dict)])
        
        # 使用新的AI知识图谱转换器
        return self.converter.text_to_triplets(all_text, query)

    def _create_knowledge_graph_from_triplets(self, triplets: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """从三元组创建知识图谱"""
        kg_graph = self.converter.create_knowledge_graph(triplets)
        
        # 转换为可序列化的格式
        kg_data = {
            'nodes': [],
            'edges': []
        }
        
        for node in kg_graph.nodes():
            kg_data['nodes'].append({
                'id': node,
                'label': node,
                'type': kg_graph.nodes[node].get('type', 'entity')
            })
        
        for edge in kg_graph.edges(data=True):
            kg_data['edges'].append({
                'from': edge[0],
                'to': edge[1],
                'relation': edge[2].get('relation', ''),
                'label': edge[2].get('label', '')
            })
        
        return kg_data

    def _analyze_impact_with_context(self, query: str, timeline: List[Dict[str, Any]], kg_data: Dict[str, Any], probabilities: Dict[str, float]) -> str:
        """基于时间线和知识图谱上下文进行影响分析"""

        
        # 准备三元组信息
        triplets_info = ""
        if 'edges' in kg_data:
            for edge in kg_data['edges'][:30]:  # 增加数量以支持更多场景
                triplets_info += f"({edge['from']}, {edge['relation']}, {edge['to']})\n"
        
        prompt = f"""
        Based on the following complete information for analysis:

        Original query: {query}
        Probability analysis results: {json.dumps(probabilities, ensure_ascii=False)}
        Knowledge graph information: {json.dumps(kg_data, ensure_ascii=False)}
        Timeline information:
        {timeline}
        
        Knowledge graph triplets:
        {triplets_info}
        
        Please provide detailed impact prediction analysis in the following format:

        ### Prediction Analysis Results
        #### Comprehensive Assessment Based on Timeline and Knowledge Graph

        ### Short-term Predictions (Next 1-6 months)
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing timeline events and triplets) |
        |-------------|-------------|---------------------|-------------|--------------------------------------------|
        | 1 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite specific timeline events and related triplets] |
        | 2 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite specific timeline events and related triplets] |
        | 3 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite specific timeline events and related triplets] |
        | 4 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite specific timeline events and related triplets] |
        | 5 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite specific timeline events and related triplets] |
        | 6 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite specific timeline events and related triplets] |
        | 7 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite specific timeline events and related triplets] |
        | 8 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite specific timeline events and related triplets] |

        ### Medium-term Predictions (Next 6 months - 2 years)
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing timeline trends and triplet relationships) |
        |-------------|-------------|---------------------|-------------|---------------------------------------------------------|
        | 1 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite timeline trends and related triplet relationships] |
        | 2 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite timeline trends and related triplet relationships] |
        | 3 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite timeline trends and related triplet relationships] |
        | 4 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite timeline trends and related triplet relationships] |
        | 5 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite timeline trends and related triplet relationships] |
        | 6 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite timeline trends and related triplet relationships] |
        | 7 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite timeline trends and related triplet relationships] |
        | 8 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite timeline trends and related triplet relationships] |

        ### Long-term Predictions (Beyond 2 years)
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing structural changes and triplet patterns) |
        |-------------|-------------|---------------------|-------------|-------------------------------------------------------|
        | 1 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite structural changes and triplet patterns] |
        | 2 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite structural changes and triplet patterns] |
        | 3 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite structural changes and triplet patterns] |
        | 4 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite structural changes and triplet patterns] |
        | 5 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite structural changes and triplet patterns] |
        | 6 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite structural changes and triplet patterns] |
        | 7 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite structural changes and triplet patterns] |
        | 8 | [Specific scenario description] | [Trigger event/factor] | [Probability]% | [Cite structural changes and triplet patterns] |

        Requirements:
        1. Provide 5-8 main scenarios for each time period (minimum 5, maximum 8)
        2. Probability should be calculated based on timeline event frequency and knowledge graph relationship strength
        3. Clearly cite specific timeline events and knowledge graph triplets as basis
        4. Total probability for each time period should be 100%
        5. Respond in English
        6. Ensure scenario descriptions are specific, detailed, and have practical basis
        """
        
        response = self.llm.invoke(prompt)
        return response.content

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query"""
        prompt = f"""
        Extract key entity names from the following query:
        Query: {query}
        
        Return entity names in JSON array format, e.g.: ["entity1", "entity2", "entity3"]
        """
        
        response = self.llm.invoke(prompt)
        return json.loads(response.content)

    def _generate_report(self, query: str, probabilities: Dict[str, float], kg_info: Dict) -> str:
        """Generate probability analysis report"""
        prompt = f"""
        Based on the following information, generate a detailed analysis report:
        
        Original query: {query}
        Probability analysis results: {json.dumps(probabilities, ensure_ascii=False)}
        Knowledge graph information: {json.dumps(kg_info, ensure_ascii=False)}
        
        Please provide a detailed probability analysis report in the following format:

        ### Short-term (next 1-6 months)
        #### Step 1 & 2: Possible scenarios and probabilities
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing triplets) |
        |----------|------|--------------|------|-------------------|
        | 1 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 2 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 3 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 4 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 5 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 6 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 7 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 8 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |

        ### Medium-term (6 months - 2 years)
        #### Step 1 & 2: Possible scenarios and probabilities
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing triplets) |
        |----------|------|--------------|------|-------------------|
        | 1 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 2 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 3 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 4 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 5 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 6 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 7 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 8 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |

        ### Long-term (2+ years)
        #### Step 1 & 2: Possible scenarios and probabilities
        | Scenario ID | Description | Key Trigger Factors | Probability | Basis (citing triplets) |
        |----------|------|--------------|------|-------------------|
        | 1 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 2 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 3 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 4 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 5 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 6 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 7 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |
        | 8 | [Scenario description] | [Trigger factors] | [Probability]% | [Cite triplets and explanation] |

        Requirements:
        1. Provide 5-8 main scenarios for each time period (minimum 5, maximum 8)
        2. Total probability for each time period should be 100%
        3. Clearly cite knowledge graph triplets as basis
        4. Scenario descriptions should be specific, detailed, and have practical basis
        5. Respond in English
        """
        
        response = self.llm.invoke(prompt)
        return response.content


class MultiAgentWorkflowSystem:
    """多Agent工作流系统"""
    
    def __init__(self, api_key: str, api_version: str, azure_endpoint: str, cue: str ,use_deepseek: bool = True):
        # 初始化LLM - 支持DeepSeek和Azure OpenAI

        # 使用DeepSeek API
        self.llm: ChatOpenAI = ChatOpenAI(
            api_key=api_key,  # 使用DeepSeek API key
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat"
        )
        # 初始化工具
        self.search_tool = TavilySearchTool()
        self.kg_tool = KnowledgeGraphTool()
        self.prob_tool = ProbabilityCalculatorTool()
        self.cue = cue
        # 初始化Agent
        self.supervisor = SupervisorAgent(self.llm)
        self.workflow1 = Workflow1Agent(self.llm, self.search_tool, self.kg_tool,self.prob_tool)
        #self.workflow2 = Workflow2Agent(self.llm, self.prob_tool, self.kg_tool)
    
    def _format_timeline_summary(self, timeline_data: List[Dict]) -> str:
        """Format timeline information summary"""
        if not timeline_data:
            return ""
        
        summary = "📅 **Timeline Information Summary:**\n\n"
        
        # Group by date
        timeline_by_date = {}
        for item in timeline_data:
            date = item.get('date', 'Unknown date')
            if date not in timeline_by_date:
                timeline_by_date[date] = []
            timeline_by_date[date].append(item)
        
        # Generate summary
        for date, events in timeline_by_date.items():
            summary += f"**{date}**:\n"
            for event in events[:3]:  # Maximum 3 events per date
                title = event.get('title', 'No title')
                content_preview = event.get('content', '')[:50] + '...' if len(event.get('content', '')) > 50 else event.get('content', '')
                summary += f"  • {title}: {content_preview}\n"
            if len(events) > 3:
                summary += f"  There are {len(events) - 3} more related events...\n"
            summary += "\n"
        
        return summary

    def _format_probability_summary(self, probabilities: Dict[str, float]) -> str:
        """Format probability information summary"""
        if not probabilities:
            return ""
        
        summary = "📊 **Probability Analysis Summary:**\n\n"
        
        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for scenario, prob in sorted_probs:
            percentage = prob * 100
            summary += f"• {scenario}: {percentage:.1f}%\n"
        
        return summary

    def process_query(self, query: str, status_callback=None) -> Dict[str, Any]:
        """处理用户查询"""
        # 分析意图
        if status_callback:
            status_callback("Analysing user intention to determine workflow")
        intent = self.supervisor.analyze_intent(query)
        st.session_state.intent = intent

        # 根据意图路由到对应工作流
        if intent == 'workflow1':
            if status_callback:
                status_callback("Processing with Workflow1: Event impact analysis")
            raw_result = self.workflow1.process(query, status_callback)
            
            # 格式化响应，整合时间线信息，不输出原始搜索结果和知识图谱
            timeline_summary = self._format_timeline_summary(raw_result.get('timeline', []))
            impact_analysis = raw_result.get('impact_analysis', '')
            
            formatted_response = f"{impact_analysis}\n\n{timeline_summary}"
            
            result = {
                'workflow_type': 'workflow1',
                'response': formatted_response,
                'timeline': raw_result.get('timeline', []),
                'knowledge_graph': raw_result.get('knowledge_graph', {}),
                'search_results': raw_result.get('search_results', [])
            }

        else:
            # 正常聊天处理
            if status_callback:
                status_callback("Processing with normal chat mode")

            prompt_template = ChatPromptTemplate.from_messages([
                                ("system", self.cue),
                                ("human", "{query}")
                            ])
            prompt = prompt_template.format(query=query)
            response = self.llm.invoke(prompt)
            
            result = {
                'workflow_type': 'normal',
                'response': response.content
            }
        
        return result

# Streamlit可视化工具
