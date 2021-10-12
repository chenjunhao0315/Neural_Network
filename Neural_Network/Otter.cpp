//
//  Otter.cpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/10/9.
//

#include "Otter.hpp"

string parse_arg(fstream &f) {
    string arg;
    f >> arg;
    if (arg[0] == '"') {
        if (arg[arg.size() - 1] != '"') {
            string append;
            getline(f, append, '"');
            arg.append(append);
        }
    }
    return arg;
}

Option parse_option(fstream &f) {
    size_t mark;
    string type;
    string arg;
    f >> type;
    if (f.eof()) return Option("End", type);
    if ((mark = type.find(':')) != string::npos) {
        if (mark == type.size() - 1) {
            type = type.substr(0, mark);
            arg = parse_arg(f);
        } else {
            arg = type.substr(mark + 1);
            type = type.substr(0, mark);
        }
    } else if ((mark = type.find('{')) != string::npos) {
        type = type.substr(0, mark);
        EARSE_SPACE(type);
        return Option("Partner", type);
    } else if ((mark = type.find('}')) != string::npos) {
        return Option("End", type);
    } else if (type[0] == '#') {
        getline(f, type, '\n');
        return Option("Comment", type);
    } else if (type[0] == '$') {
        return Option("End", "End of otter syntax");
    } else {
        string find_colon;
        f >> find_colon;
        if ((mark = find_colon.find('{')) != string::npos) {
            EARSE_SPACE(type);
            return Option("Partner", type);
        } else if ((mark = find_colon.find(':')) == string::npos) {
            fprintf(stderr, "[Stick] Syntax error!\n");
        } else {
            if (mark == find_colon.size() - 1) {
                arg = parse_arg(f);
            } else {
                arg = find_colon.substr(mark + 1);
            }
        }
    }
    
    EARSE_SPACE(type);
    EARSE_CHARACTER(arg, '"');
    
    return Option(type, arg);
}

bool Otter_Leader::read_project(const char *project_file) {
    fstream project;
    project.open(project_file);
    if (!project.is_open()) {
        fprintf(stderr, "[Otter_Leader] Open file fail!\n");
        exit(-1);
    }
    
    Stick project_title = parse_option(project);
    if (project_title.type == "name")
        project_name = project_title.info;
    else
        fprintf(stderr, "[Otter_Leader] Syntax error!\n");
    
    Option segment = parse_option(project);
    while (segment.type != "End") {
        if (segment.type == "Partner") {
            Otter team_leader(segment.info);
            team_leader.parse_blueprint(project);
            teams.push_back(team_leader);
        } else if (segment.type == "Comment") {
            // skip
        } else {
            option.push_back(segment);
        }
        segment = parse_option(project);
    }
    
    project.close();
    return true;
}

bool Otter_Leader::save_project(const char *project_file) {
    FILE *project = fopen(project_file, "w");
    if (!project) return false;
    
    fprintf(project, "name: \"%s\"\n", project_name.c_str());
    for (int i = 0; i < option.size(); ++i) {
        if (option[i].info.find(' ') == string::npos)
            fprintf(project, "%s: %s\n", option[i].type.c_str(), option[i].info.c_str());
        else
            fprintf(project, "%s: \"%s\"\n", option[i].type.c_str(), option[i].info.c_str());
    }
    
    for (int i = 0; i < teams.size(); ++i) {
        teams[i].save_blueprint(project);
    }
    
    fclose(project);
    return true;
}

bool Otter_Leader::save_raw(const char *project_file) {
    FILE *project = fopen(project_file, "w");
    
    fprintf(project, "name: \"%s\" ", project_name.c_str());
    for (int i = 0; i < option.size(); ++i) {
        if (option[i].info.find(' ') == string::npos)
            fprintf(project, "%s: %s ", option[i].type.c_str(), option[i].info.c_str());
        else
            fprintf(project, "%s: \"%s\" ", option[i].type.c_str(), option[i].info.c_str());
    }
    for (int i = 0; i < teams.size(); ++i) {
        teams[i].save_raw(project);
    }
    fprintf(project, "$\n");
    fclose(project);
    return true;
}

bool Otter::parse_blueprint(fstream &blueprint) {
    Stick element = parse_option(blueprint);
    while (element.type != "End") {
        if (element.type == "Partner") {
            Otter team_leader(element.info);
            team_leader.parse_blueprint(blueprint);
            partner.push_back(team_leader);
        } else if (element.type == "Comment") {
            // skip
        } else {
            material.push_back(Stick(element.type, element.info));
        }
        element = parse_option(blueprint);
    }
    return true;
}

void Otter::save_blueprint(FILE *project, int format) {
    WRITE_SPACE(project, format);
    fprintf(project, "%s {\n", name.c_str());
    
    for (int i = 0; i < material.size(); ++i) {
        WRITE_SPACE(project, (format + 4));
        if (material[i].info.find(' ') != string::npos)
            fprintf(project, "%s: \"%s\"\n", material[i].type.c_str(), material[i].info.c_str());
        else
            fprintf(project, "%s: %s\n", material[i].type.c_str(), material[i].info.c_str());
    }
    
    for (int i = 0; i < partner.size(); ++i) {
        partner[i].save_blueprint(project, format + 4);
    }
    
    WRITE_SPACE(project, format);
    fprintf(project, "}\n");
}

void Otter::save_raw(FILE *project) {
    fprintf(project, "%s { ", name.c_str());
    
    for (int i = 0; i < material.size(); ++i) {
        if (material[i].info.find(' ') != string::npos)
            fprintf(project, "%s: \"%s\" ", material[i].type.c_str(), material[i].info.c_str());
        else
            fprintf(project, "%s: %s ", material[i].type.c_str(), material[i].info.c_str());
    }
    
    for (int i = 0; i < partner.size(); ++i) {
        partner[i].save_raw(project);
    }

    fprintf(project, "} ");
}

vector<Stick> Otter::getMaterial() {
    vector<Stick> materials;
    materials.insert(materials.end(), material.begin(), material.end());
    
    for (int i = 0; i < partner.size(); ++i) {
        vector<Stick> material = partner[i].getMaterial();
        materials.insert(materials.end(), material.begin(), material.end());
    }
    
    return materials;
}

vector<Stick> Otter_Leader::getMaterial(int index) {
    if (index > teams.size())
        return vector<Stick>();
    return teams[index].getMaterial();
}
