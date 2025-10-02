#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
내부 문서 RAG 성능 테스트
- 검색 속도 측정
- 답변 정확도 평가
- 시각화 결과 생성
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 상위 디렉토리를 Python 경로에 추가
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# matplotlib 임포트 (선택적)
try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    HAS_MATPLOTLIB = True
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib가 설치되지 않아 시각화 기능을 사용할 수 없습니다.")

@dataclass
class PerformanceMetrics:
    """성능 측정 결과"""
    query: str
    answer: str            # 생성된 답변 내용
    retrieval_time: float  # 검색 시간 (초)
    total_time: float      # 총 응답 시간 (초)
    answer_length: int     # 답변 길이
    accuracy_score: Optional[float] = None      # 키워드 기반 정확도 점수 (0-10)
    llm_accuracy_score: Optional[float] = None  # LLM 기반 정확도 점수 (0-10)
    embedding_similarity: Optional[float] = None # 임베딩 유사도 점수 (0-1)
    final_accuracy_score: Optional[float] = None # 종합 정확도 점수 (0-10)
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class RAGPerformanceTester:
    """RAG 성능 테스터"""
    
    def __init__(self):
        self.results: List[PerformanceMetrics] = []
        self.retrieve_documents = None
        self.generate_answer = None
        
    def import_services(self):
        """RAG 서비스 임포트"""
        try:
            # 이모지 출력 문제를 피하기 위해 stdout을 임시로 리다이렉트
            import io
            import contextlib
            
            # 임포트 시 출력 억제
            with contextlib.redirect_stdout(io.StringIO()):
                from internal_retrieve import RetrieveService
            
            # 서비스 인스턴스 생성
            self.rag_service = RetrieveService()
            
            # 래퍼 함수 생성 (출력 억제)
            def safe_retrieve_documents(query, top_k=5):
                with contextlib.redirect_stdout(io.StringIO()):
                    return self.rag_service.retrieve_documents(query, top_k)
            
            def safe_generate_answer(query, contexts):
                with contextlib.redirect_stdout(io.StringIO()):
                    return self.rag_service.generate_answer(query, contexts)
            
            self.retrieve_documents = safe_retrieve_documents
            self.generate_answer = safe_generate_answer
            
            print("RAG 서비스 초기화 성공")
            return True
        except Exception as e:
            error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
            print(f"RAG 서비스 import 실패: {error_msg}")
            return False
    
    def get_test_queries(self) -> List[str]:
        """테스트 쿼리 목록 반환"""
        return [
            "생산현황 작성 시 주의사항 알려줘",
            "회사에서 브이로그 찍어 유튜브에 올리려고 하는데 회사 규정에 어긋나는지 알려줘",
            "직원의 연차휴가는 몇 일인가요?",
            "기록물 관리 절차에 대해 설명해주세요",
            "인사평가는 언제 실시하나요?",
            "문서 보관 기간은 얼마나 되나요?",
            "직원 교육 프로그램이 있나요?",
            "기록물의 보존기간은 어떻게 정해지나요?",
            "문서 분류 기준은 무엇인가요?",
            "근무시간은 어떻게 정해져 있나요?"
        ]
    
    def evaluate_keyword_accuracy(self, query: str, answer: str) -> float:
        """키워드 기반 정확도 평가"""
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        # 키워드 매칭 기반 간단한 평가
        query_keywords = set(query.lower().replace('?', '').replace('.', '').split())
        answer_keywords = set(answer.lower().replace('?', '').replace('.', '').split())
        
        # 공통 키워드 비율
        common_keywords = query_keywords.intersection(answer_keywords)
        if not query_keywords:
            return 5.0  # 기본 점수
        
        keyword_score = len(common_keywords) / len(query_keywords)
        
        # 답변 길이 보정 (너무 짧거나 긴 답변 페널티)
        length_score = 1.0
        if len(answer) < 50:
            length_score = 0.5
        elif len(answer) > 1000:
            length_score = 0.8
        
        # 최종 점수 (0-10 스케일)
        final_score = min(10.0, (keyword_score * 0.7 + length_score * 0.3) * 10)
        return round(final_score, 2)
    
    def evaluate_llm_accuracy(self, query: str, answer: str) -> Optional[float]:
        """LLM 기반 정확도 평가 (GPT 사용)"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            # OpenAI API 키 확인
            if not os.getenv("OPENAI_API_KEY"):
                print("  LLM 평가 건너뜀: OPENAI_API_KEY 없음")
                return None
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                당신은 RAG 시스템의 답변 품질을 평가하는 전문가입니다.
                주어진 질문과 답변을 분석하여 답변의 품질을 0-10점으로 평가하세요.
                
                평가 기준:
                1. 정확성 (40%): 답변이 질문에 정확히 대답하는가?
                2. 완성도 (30%): 답변이 충분히 상세하고 완전한가?
                3. 관련성 (20%): 답변이 질문과 관련이 있는가?
                4. 명확성 (10%): 답변이 이해하기 쉽고 명확한가?
                
                점수만 숫자로 반환하세요 (예: 8.5).
                """),
                ("user", "질문: {question}\n\n답변: {answer}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"question": query, "answer": answer})
            
            # 숫자 추출
            import re
            score_match = re.search(r'(\d+\.?\d*)', result.strip())
            if score_match:
                score = float(score_match.group(1))
                return min(10.0, max(0.0, score))  # 0-10 범위로 제한
            else:
                print(f"  LLM 평가 파싱 실패: {result}")
                return None
                
        except Exception as e:
            print(f"  LLM 평가 오류: {e}")
            return None
    
    def evaluate_embedding_similarity(self, query: str, answer: str) -> Optional[float]:
        """임베딩 유사도 평가"""
        try:
            from langchain_openai import OpenAIEmbeddings
            import numpy as np
            
            # OpenAI API 키 확인
            if not os.getenv("OPENAI_API_KEY"):
                print("  임베딩 평가 건너뜀: OPENAI_API_KEY 없음")
                return None
            
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # 질문과 답변의 임베딩 생성
            query_embedding = embeddings.embed_query(query)
            answer_embedding = embeddings.embed_query(answer)
            
            # 코사인 유사도 계산
            query_vec = np.array(query_embedding)
            answer_vec = np.array(answer_embedding)
            
            similarity = np.dot(query_vec, answer_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(answer_vec)
            )
            
            return float(similarity)  # 0-1 범위
            
        except Exception as e:
            print(f"  임베딩 평가 오류: {e}")
            return None
    
    def calculate_final_accuracy(self, keyword_score: float, llm_score: Optional[float], 
                               embedding_score: Optional[float]) -> float:
        """종합 정확도 점수 계산"""
        scores = []
        weights = []
        
        # 키워드 점수 (기본)
        scores.append(keyword_score)
        weights.append(0.3)
        
        # LLM 점수
        if llm_score is not None:
            scores.append(llm_score)
            weights.append(0.5)
        
        # 임베딩 점수 (0-1을 0-10으로 변환)
        if embedding_score is not None:
            scores.append(embedding_score * 10)
            weights.append(0.2)
        
        # 가중 평균 계산
        if not scores:
            return 0.0
        
        # 가중치 정규화
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        final_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        return round(final_score, 2)
    
    def test_single_query(self, query: str) -> PerformanceMetrics:
        """단일 쿼리 성능 테스트"""
        print(f"테스트 중: {query}")
        
        try:
            # 전체 시간 측정 시작
            total_start_time = time.time()
            
            # 문서 검색 시간 측정
            retrieval_start_time = time.time()
            contexts = self.retrieve_documents(query, top_k=5)
            retrieval_time = time.time() - retrieval_start_time
            
            # 답변 생성
            answer = self.generate_answer(query, contexts)
            total_time = time.time() - total_start_time
            
            # 다중 정확도 평가
            print(f"  정확도 평가 중...")
            
            # 1. 키워드 기반 정확도
            keyword_accuracy = self.evaluate_keyword_accuracy(query, answer)
            print(f"    키워드 정확도: {keyword_accuracy:.2f}/10")
            
            # 2. LLM 기반 정확도
            llm_accuracy = self.evaluate_llm_accuracy(query, answer)
            if llm_accuracy is not None:
                print(f"    LLM 정확도: {llm_accuracy:.2f}/10")
            
            # 3. 임베딩 유사도
            embedding_similarity = self.evaluate_embedding_similarity(query, answer)
            if embedding_similarity is not None:
                print(f"    임베딩 유사도: {embedding_similarity:.3f}")
            
            # 4. 종합 정확도
            final_accuracy = self.calculate_final_accuracy(
                keyword_accuracy, llm_accuracy, embedding_similarity
            )
            print(f"    종합 정확도: {final_accuracy:.2f}/10")
            
            metrics = PerformanceMetrics(
                query=query,
                answer=answer if answer else "",
                retrieval_time=retrieval_time,
                total_time=total_time,
                answer_length=len(answer) if answer else 0,
                accuracy_score=keyword_accuracy,
                llm_accuracy_score=llm_accuracy,
                embedding_similarity=embedding_similarity,
                final_accuracy_score=final_accuracy
            )
            
            print(f"  검색: {retrieval_time:.2f}s, 총: {total_time:.2f}s")
            print(f"  답변 길이: {metrics.answer_length}자")
            
            return metrics
            
        except Exception as e:
            error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
            print(f"  오류 발생: {error_msg}")
            return PerformanceMetrics(
                query=query,
                answer="",
                retrieval_time=0,
                total_time=0,
                answer_length=0,
                accuracy_score=0,
                llm_accuracy_score=None,
                embedding_similarity=None,
                final_accuracy_score=0
            )
    
    def run_performance_test(self) -> List[PerformanceMetrics]:
        """성능 테스트 실행"""
        if not self.import_services():
            print("RAG 서비스를 사용할 수 없어 테스트를 중단합니다.")
            return []
        
        queries = self.get_test_queries()
        results = []
        
        print(f"\n성능 테스트 시작 - 총 {len(queries)}개 쿼리")
        print("=" * 60)
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}]", end=" ")
            result = self.test_single_query(query)
            results.append(result)
        
        self.results.extend(results)
        
        # 요약 통계 출력
        self._print_summary_stats(results)
        
        return results
    
    def _print_summary_stats(self, results: List[PerformanceMetrics]):
        """요약 통계 출력"""
        if not results:
            return
        
        retrieval_times = [r.retrieval_time for r in results if r.retrieval_time > 0]
        total_times = [r.total_time for r in results if r.total_time > 0]
        keyword_scores = [r.accuracy_score for r in results if r.accuracy_score is not None and r.accuracy_score > 0]
        llm_scores = [r.llm_accuracy_score for r in results if r.llm_accuracy_score is not None]
        embedding_scores = [r.embedding_similarity for r in results if r.embedding_similarity is not None]
        final_scores = [r.final_accuracy_score for r in results if r.final_accuracy_score is not None and r.final_accuracy_score > 0]
        
        print("\n" + "=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)
        
        if retrieval_times:
            avg_retrieval = sum(retrieval_times) / len(retrieval_times)
            print(f"평균 검색 시간: {avg_retrieval:.2f}초")
        
        if total_times:
            avg_total = sum(total_times) / len(total_times)
            print(f"평균 총 응답 시간: {avg_total:.2f}초")
        
        print("\n정확도 분석:")
        if keyword_scores:
            avg_keyword = sum(keyword_scores) / len(keyword_scores)
            print(f"  키워드 기반 정확도: {avg_keyword:.2f}/10")
        
        if llm_scores:
            avg_llm = sum(llm_scores) / len(llm_scores)
            print(f"  LLM 기반 정확도: {avg_llm:.2f}/10")
        
        if embedding_scores:
            avg_embedding = sum(embedding_scores) / len(embedding_scores)
            print(f"  임베딩 유사도: {avg_embedding:.3f}")
        
        if final_scores:
            avg_final = sum(final_scores) / len(final_scores)
            print(f"  종합 정확도: {avg_final:.2f}/10")
        else:
            print("  정확도 데이터: 없음")
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """결과를 JSON 파일로 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_results_{timestamp}.json"
        
        # test 폴더에 저장
        save_path = Path(__file__).parent / filename
        
        results_data = [asdict(result) for result in self.results]
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"결과 저장 완료: {save_path}")
        return str(save_path)
    
    def create_performance_chart(self, results: List[PerformanceMetrics], save_path: Optional[str] = None):
        """성능 차트 생성"""
        if not HAS_MATPLOTLIB:
            print("matplotlib가 없어 시각화를 생성할 수 없습니다.")
            return
        
        if not results:
            print("시각화할 데이터가 없습니다.")
            return
        
        # 데이터 준비
        queries = [r.query[:30] + "..." if len(r.query) > 30 else r.query for r in results]
        retrieval_times = [r.retrieval_time for r in results]
        total_times = [r.total_time for r in results]
        keyword_scores = [r.accuracy_score if r.accuracy_score else 0 for r in results]
        llm_scores = [r.llm_accuracy_score if r.llm_accuracy_score else 0 for r in results]
        embedding_scores = [r.embedding_similarity if r.embedding_similarity else 0 for r in results]
        final_scores = [r.final_accuracy_score if r.final_accuracy_score else 0 for r in results]
        
        # 차트 생성 (2x4 레이아웃으로 확장)
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('RAG 성능 테스트 결과 (다중 정확도 평가)', fontsize=16, fontweight='bold')
        
        # 1. 검색 시간 막대 그래프
        ax1.bar(range(len(queries)), retrieval_times, color='skyblue', alpha=0.7)
        ax1.set_title('검색 시간 (초)', fontweight='bold')
        ax1.set_xlabel('쿼리')
        ax1.set_ylabel('시간 (초)')
        ax1.set_xticks(range(len(queries)))
        ax1.set_xticklabels(queries, rotation=45, ha='right')
        
        # 2. 총 응답 시간 막대 그래프
        ax2.bar(range(len(queries)), total_times, color='lightcoral', alpha=0.7)
        ax2.set_title('총 응답 시간 (초)', fontweight='bold')
        ax2.set_xlabel('쿼리')
        ax2.set_ylabel('시간 (초)')
        ax2.set_xticks(range(len(queries)))
        ax2.set_xticklabels(queries, rotation=45, ha='right')
        
        # 3. 키워드 기반 정확도
        ax3.bar(range(len(queries)), keyword_scores, color='lightgreen', alpha=0.7)
        ax3.set_title('키워드 기반 정확도 (0-10)', fontweight='bold')
        ax3.set_xlabel('쿼리')
        ax3.set_ylabel('점수')
        ax3.set_xticks(range(len(queries)))
        ax3.set_xticklabels(queries, rotation=45, ha='right')
        ax3.set_ylim(0, 10)
        
        # 4. LLM 기반 정확도
        ax4.bar(range(len(queries)), llm_scores, color='gold', alpha=0.7)
        ax4.set_title('LLM 기반 정확도 (0-10)', fontweight='bold')
        ax4.set_xlabel('쿼리')
        ax4.set_ylabel('점수')
        ax4.set_xticks(range(len(queries)))
        ax4.set_xticklabels(queries, rotation=45, ha='right')
        ax4.set_ylim(0, 10)
        
        # 5. 임베딩 유사도
        ax5.bar(range(len(queries)), embedding_scores, color='orange', alpha=0.7)
        ax5.set_title('임베딩 유사도 (0-1)', fontweight='bold')
        ax5.set_xlabel('쿼리')
        ax5.set_ylabel('유사도')
        ax5.set_xticks(range(len(queries)))
        ax5.set_xticklabels(queries, rotation=45, ha='right')
        ax5.set_ylim(0, 1)
        
        # 6. 종합 정확도
        ax6.bar(range(len(queries)), final_scores, color='mediumpurple', alpha=0.7)
        ax6.set_title('종합 정확도 (0-10)', fontweight='bold')
        ax6.set_xlabel('쿼리')
        ax6.set_ylabel('점수')
        ax6.set_xticks(range(len(queries)))
        ax6.set_xticklabels(queries, rotation=45, ha='right')
        ax6.set_ylim(0, 10)
        
        # 7. 시간 성능 요약
        valid_retrieval = [t for t in retrieval_times if t > 0]
        valid_total = [t for t in total_times if t > 0]
        
        time_categories = []
        time_values = []
        time_colors = []
        
        if valid_retrieval:
            time_categories.append('평균 검색시간\n(초)')
            time_values.append(sum(valid_retrieval) / len(valid_retrieval))
            time_colors.append('skyblue')
        
        if valid_total:
            time_categories.append('평균 총시간\n(초)')
            time_values.append(sum(valid_total) / len(valid_total))
            time_colors.append('lightcoral')
        
        if time_categories:
            ax7.bar(time_categories, time_values, color=time_colors, alpha=0.7)
            ax7.set_title('시간 성능 요약', fontweight='bold')
            ax7.set_ylabel('시간 (초)')
            
            # 값 표시
            for i, v in enumerate(time_values):
                ax7.text(i, v + max(time_values) * 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax7.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=ax7.transAxes, fontsize=14)
            ax7.set_title('시간 성능 요약', fontweight='bold')
        
        # 8. 정확도 성능 요약
        valid_keyword = [s for s in keyword_scores if s > 0]
        valid_llm = [s for s in llm_scores if s > 0]
        valid_embedding = [s for s in embedding_scores if s > 0]
        valid_final = [s for s in final_scores if s > 0]
        
        accuracy_categories = []
        accuracy_values = []
        accuracy_colors = []
        
        if valid_keyword:
            accuracy_categories.append('키워드\n(0-10)')
            accuracy_values.append(sum(valid_keyword) / len(valid_keyword))
            accuracy_colors.append('lightgreen')
        
        if valid_llm:
            accuracy_categories.append('LLM\n(0-10)')
            accuracy_values.append(sum(valid_llm) / len(valid_llm))
            accuracy_colors.append('gold')
        
        if valid_embedding:
            accuracy_categories.append('임베딩\n(0-1)')
            accuracy_values.append(sum(valid_embedding) / len(valid_embedding))
            accuracy_colors.append('orange')
        
        if valid_final:
            accuracy_categories.append('종합\n(0-10)')
            accuracy_values.append(sum(valid_final) / len(valid_final))
            accuracy_colors.append('mediumpurple')
        
        if accuracy_categories:
            ax8.bar(accuracy_categories, accuracy_values, color=accuracy_colors, alpha=0.7)
            ax8.set_title('정확도 성능 요약', fontweight='bold')
            ax8.set_ylabel('점수')
            
            # 값 표시
            for i, v in enumerate(accuracy_values):
                ax8.text(i, v + max(accuracy_values) * 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax8.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=ax8.transAxes, fontsize=14)
            ax8.set_title('정확도 성능 요약', fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(__file__).parent / f"performance_chart_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"성능 차트 저장: {save_path}")
        plt.close()

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("내부 문서 RAG 성능 테스트")
    print("=" * 60)
    
    tester = RAGPerformanceTester()
    results = tester.run_performance_test()
    
    if results:
        print(f"\n테스트 완료! 총 {len(results)}개 쿼리 처리됨")
        
        # 결과 저장
        tester.save_results()
        
        # 시각화 생성
        if HAS_MATPLOTLIB:
            print("\n시각화 생성 중...")
            tester.create_performance_chart(results)
        
    else:
        print("테스트 실행 실패!")

if __name__ == "__main__":
    main()
