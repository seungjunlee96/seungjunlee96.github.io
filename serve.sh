#!/bin/bash

# Jekyll 블로그 로컬 서버 실행 스크립트
# Usage: ./serve.sh

set -e  # 에러 발생 시 스크립트 중지

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 헤더 출력
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}   Jekyll Blog Local Server${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Ruby 설치 확인
echo -e "${YELLOW}[1/3] Checking Ruby installation...${NC}"
if command -v ruby &> /dev/null; then
    RUBY_VERSION=$(ruby -v)
    echo -e "${GREEN}✓ Ruby found: $RUBY_VERSION${NC}"
else
    echo -e "${RED}✗ Ruby is not installed.${NC}"
    echo "Please install Ruby first:"
    echo "  brew install ruby"
    exit 1
fi
echo ""

# Bundler 설치 확인
echo -e "${YELLOW}[2/3] Checking Bundler installation...${NC}"
if command -v bundle &> /dev/null; then
    echo -e "${GREEN}✓ Bundler found${NC}"
else
    echo -e "${YELLOW}Installing Bundler...${NC}"
    gem install bundler
fi
echo ""

# 의존성 설치
echo -e "${YELLOW}[3/3] Installing dependencies...${NC}"
bundle install
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Jekyll 서버 실행
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}Starting Jekyll server...${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${YELLOW}The site will be available at:${NC}"
echo -e "${GREEN}  http://localhost:4000${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Jekyll 서버 실행
bundle exec jekyll serve --host 0.0.0.0 --port 4000 --livereload

