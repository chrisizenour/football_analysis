# NFL SALARY CAP ANALYSIS LITERATURE REVIEW

## CHRISTOPHER IZENOUR, MS ANALYTICS

# Purpose

This literature review examines research relevant to optimal salary cap allocations in the National Football League (NFL). The review synthesizes studies on salary cap (scarce resource) allocation and market efficiency, with a focus on their methodologies, findings, and gaps. This project seeks to understand what the optimal NFL team salary cap allocation is based on 14 seasons’ worth of data (2011–2024).

# Background and Context

The NFL’s hard salary cap, introduced in 1994 as part of the NFL–NFL Players’ Association (NFLPA) Collective Bargaining Agreement (CBA), limits team spending on player salaries to promote competitive balance (Borghesi, 2008; Spotrac, n.d.). The cap was $34.6M in 1994, $120.6M in 2011, and $255.4M in 2024, with an uncapped year in 2010 (Spotrac, n.d.). Over 31 years, the cap has shaped roster strategies, offering a rich context for studying resource allocation (Borghesi, 2008; Larsen et al., 2006). Recent CBAs (e.g., 2011, 2020) introduced expanded rookie contract structures and cap flexibility, influencing allocation strategies.

Coincident with the salary cap’s evolution, data science and artificial intelligence/machine learning (AI/ML) have transformed sports analytics (Morgulev et al., 2018). Teams leverage computational advancements to analyze gameplay, player selection, and front-office decisions. As compute and storage costs decline and processing power increases, teams adopting data-driven approaches gain a competitive edge. This review integrates insights from economic, statistical, and organizational perspectives to guide the development of an updated allocation model.

# Thematic Analysis

## Salary Cap Allocation Strategies

Research on NFL salary cap allocation explores how teams distribute funds across positions, between current and new players, through incentives to maximize wins.

### *Leeds and Kowalewski (2001)*

Examined the 1993 CBA’s impact on salaries for skill position players (quarterbacks, running backs, wide receivers, tight ends) using quantile regression on ~500 players’ data from 1992 (pre-CBA) and 1994 (post-CBA) (Leeds & Kowalewski, 2001). They analyzed salaries, performance metrics (e.g., passing yards, touchdowns), and controls (e.g., experience, draft status). The CBA increased income inequality, with performance metrics driving salaries post-1994, particularly for lower-paid players (.25 quantile). High-performing veterans and superstars earned disproportionately more, while rookies faced wage compression, creating a “winner-take-all” market. Persistent market inefficiencies suggest opportunities to exploit undervalued players, though the study’s focus on skill positions and two-year scope limits its applicability to modern, diverse rosters.

### *Borghesi (2008)*

Analyzed team-level salary dispersion’s impact on performance from 1994–2004 (~352 team-seasons) using OLS and Poisson regression on salary data (from USA Today Library and Research Service and NFLPA.org) and performance metrics (wins, offensive/defensive proficiency) (Borghesi, 2008). Pay dispersion (measured via Gini coefficient) was split into justified (performance-based) and unjustified (unexplained) components, with controls for payroll and coaching quality. Lower dispersion improved wins and proficiency, while high dispersion, driven by superstar structures, reduced performance due to lower-paid player dissatisfaction. Robustness tests used alternative dispersion metrics. The study advocates uniform salary structures for roster cohesion, but its team-level focus and pre-2011 data miss player-specific dynamics and recent CBA changes.

### *Mondello and Maxcy (2009)*

Studied salary dispersion and performance bonuses from 2000–2007 (~256 team-seasons) using OLS regression on salary, wins, and player performance data (Mondello & Maxcy, 2009). High salary dispersion (measured by Gini or HHI) negatively impacted wins, supporting the team cohesion hypothesis, while performance bonuses had modest positive effects, strongest for quarterbacks. Robustness tests explored alternative dispersion measures. The findings suggest balanced salaries and targeted bonuses optimize performance, but the team-level approach and outdated data limit insights into position-specific strategies and modern bonus structures.

### *Mulholland and Jensen (2019)*

Modeled predicted wins based on Approximate Value (AV), compensation, and 19 position groups from 2011–2015 (~2,500 players) using univariate regression and linear programming under the 2016 cap ($155.27M) (Mulholland & Jensen, 2019). They prioritized defensive ends (13.7%), outside linebackers (15.2%), guards (10.6%), and quarterbacks (8.6%), while allocating only 1.2% to left tackles, challenging conventional wisdom. Rookie contracts (e.g., Russell Wilson) provided “uncompensated wins,” supporting non-uniform allocation. Robustness tests used alternative AV metrics. The study highlights position-specific investments, but its market efficiency assumption and lack of team cohesion focus limit its scope.

### *Keefer (2017)*

Explored the sunk-cost fallacy in cap management from 2004–2012 (~1,000–2,000 players) using OLS regression on salaries, AV, playing time, and wins (Keefer, 2017). Retaining overpaid, underperforming players misallocated cap space, reducing wins, while cutting such players improved performance. Robustness tests used alternative performance metrics. The study emphasizes dynamic roster management, but its pre-2013 data and lack of position-specific focus limit applicability to current CBAs.

### *Shin et al. (2023)*

Though focused on Major League Baseball, offers relevant insights using GMM panel regression on 2001–2022 data (~660 team-seasons) from USA Today and Baseball-Reference.com (Shin et al., 2023). They measured human resources (HR) budget allocation to new players (New-to-Total) and pay concentration among new hires using the Herfindahl-Hirschman Index (HHI), with ex-ante performance (lagged win-to-loss ratio) as a moderator. Allocating more to new players negatively impacted performance due to lacking organization-specific skills, especially in high-performing teams, though low-performing teams benefited. Concentrated pay for star hires boosted performance but weakened and turned negative for teams with win-to-loss ratios > 1.208, possibly due to star conflicts. Robustness tests used alternative salary ratios. While the MLB salary cap is not “hard” like the NFL’s, the luxury tax disincentivizes spending prohibitive sums on salaries, suggesting teams prioritize veteran retention for successful rosters and leverage rookies for struggling ones. Limitations include MLB’s unique rules and lack of player perception data.

## Statistical and Optimization Models in Sports

Statistical and optimization models are central to NFL salary cap research. Leeds and Kowalewski (2001) used quantile regression to capture salary variations across pay levels. Borghesi (2008) and Mondello and Maxcy (2009) employed OLS and Poisson regression to quantify dispersion’s impact. Mulholland and Jensen (2019) combined univariate regression and linear programming to optimize position-specific allocations, building on economic models (Radner, 1972). Keefer (2017) used OLS to analyze sunk-cost effects, while Shin et al. (2023) applied GMM to address endogeneity. These models rely on historical data, limiting their ability to capture evolving player roles and market dynamics. Advanced AI/ML techniques, such as random forests or gradient boosting, could enhance predictive power by analyzing complex, non-linear relationships in 2011–2024 data.

# Critical Analysis

The studies offer complementary insights. Leeds and Kowalewski (2001) highlight performance-based pay post-1994, with inefficiencies suggesting undervalued player opportunities. Borghesi (2008) and Mondello and Maxcy (2009) emphasize uniform salaries for cohesion, contrasting Mulholland and Jensen’s (2019) non-uniform, position-specific approach leveraging rookie contracts. Keefer (2017) underscores sunk-cost pitfalls, aligning with Mulholland and Jensen’s focus on cap efficiency. Shin et al. (2023) add a contextual RBV perspective, suggesting allocation strategies vary by team performance. Methodologically, Mulholland and Jensen’s AV-based optimization is robust, but its market efficiency assumption is questionable given Leeds and Kowalewski’s inefficiencies. Borghesi’s team-level focus overlooks player dynamics, which Mulholland and Jensen address. Shin et al.’s MLB context requires careful NFL translation.

# Gaps and Research Opportunities

Table 1 summarizes the studies’ characteristics and limitations, highlighting gaps addressed by this project.

Most studies rely on pre-2015 data, missing the cap’s growth ($120.6M in 2011 to $255.4M in 2024), growth in inactive roster spending that counts against a team’s salary cap, and CBA changes (e.g., rookie contract structures, cap flexibility). Leeds and Kowalewski (2001) exclude defensive and line positions, while Borghesi (2008) and Mondello and Maxcy (2009) lack player-level granularity. Keefer (2017) and Mulholland and Jensen (2019) use more recent data but miss 2016–2024 trends, such as declining running back valuations. Shin et al. (2023) offer a contemporary dataset but require NFL adaptation. Traditional methods (e.g., OLS, linear programming) and public datasets (e.g., salaries, AV) limit capturing non-linear dynamics. No study uses AI/ML, which could analyze 2011–2024 data to predict winning percentages, integrating player-level (e.g., performance, injuries) and team-level (e.g., coaching) factors. Underexplored areas include defensive positions (e.g., safeties, cornerbacks), performance bonuses, sunk-cost fallacies, and RBV applications to NFL front-office expertise.

Future research should: (1) analyze 2011–2024 data, (2) employ AI/ML (e.g., clustering, random forests, gradient boosting) to uncover trends and simulate roster decisions, (3) investigate defensive position contributions in pass-heavy offenses, and (4) investigate portions of salary cap expenditures on the active/inactive rosters. These efforts will enhance data-driven NFL decision-making.

# Conclusion

The literature reveals a complex interplay of compensation, salary structures, and performance. Leeds and Kowalewski (2001) highlight performance-based pay and inefficiencies post-1994. Borghesi (2008) and Mondello and Maxcy (2009) advocate uniform salaries for cohesion, while Mulholland and Jensen (2019) propose non-uniform, position-specific allocations with rookie contracts. Keefer (2017) warns against sunk-cost fallacies, and Shin et al. (2023) suggest context-dependent allocations, favoring veterans in successful teams and stars in struggling ones. These studies underscore data-driven strategies under the salary cap. This project will build on these insights, using 2011–2024 data and AI/ML to develop an updated allocation model, addressing gaps in data scope, defensive positions, bonuses, and organizational context.

# REFERENCES

Borghesi, R. (2008). Allocation of scarce resources: Insight from the NFL salary cap. Journal of Economics and Business, 60(6), 536–550. https://doi.org/10.1016/j.jeconbus.2007.08.002

Keefer, Q. A. W. (2017). The Sunk-Cost Fallacy in the National Football League: Salary Cap Value and Playing Time. Journal of Sports Economics, 18(3), 282–297. https://doi.org/10.1177/1527002515574515

Larsen, A., Fenn, A. J., & Spenner, E. L. (2006). The Impact of Free Agency and the Salary Cap on Competitive Balance in the National Football League. Journal of Sports Economics, 7(4), 374–390. https://doi.org/10.1177/1527002505279345

Leeds, M. A., & Kowalewski, S. (2001). Winner Take All in the NFL: The Effect of the Salary Cap and Free Agency on the Compensation of Skill Position Players. Journal of Sports Economics, 2(3), 244–256. https://doi.org/10.1177/152700250100200304

Mondello, M., & Maxcy, J. (2009). The impact of salary dispersion and performance bonuses in NFL organizations. Management Decision, 47(1), 110–123. https://doi.org/10.1108/00251740910929731

Morgulev, E., Azar, O. H., & Lidor, R. (2018). Sports analytics and the big-data era. International Journal of Data Science and Analytics, 5(4), 213–222. https://doi.org/10.1007/s41060-017-0093-7

Mulholland, J., & Jensen, S. T. (2019). Optimizing the allocation of funds of an NFL team under the salary cap. International Journal of Forecasting, 35(2), 767–775. https://doi.org/10.1016/j.ijforecast.2018.09.004

Shin, H. W., Cho, S., & Lee, J. K. (2023). Performance implications of financial resource allocation in new hiring: The case of major league baseball. Management Decision, 61(10), 2829–2850. https://doi.org/10.1108/MD-07-2022-0887

Spotrac. (n.d.). NFL CBAs & Cap History. Spotrac.Com. Retrieved May 20, 2025, from https://www.spotrac.com/nfl/cba
