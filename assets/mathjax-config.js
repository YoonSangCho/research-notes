// MathJax 3 설정 — 수식 렌더링
//
// pymdownx.arithmatex(generic)는 대부분의 $...$ / $$...$$ 를
// \(...\) / \[...\] 로 변환한다. 다만 리스트 안에 들여쓰기된 한 줄짜리
// $$...$$ 블록 등 일부는 변환하지 못하고 raw 로 남는다.
// 그래서 MathJax 가 변환분(\(\),\[\])과 함께 남은 $$...$$ 까지 직접
// 인식하도록 설정한다. (코드/스크립트 등은 MathJax 기본 skipHtmlTags 로 자동 제외)
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true,
  },
};

// Material 인스턴트 내비게이션 후에도 페이지 전체 수식을 다시 렌더링
document$.subscribe(() => {
  MathJax.typesetPromise();
});
