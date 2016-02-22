import numpy as np
import logging

logger = logging.getLogger(name=__name__)


def type1_transition(sheet, edge01, epsilon=0.1):
    """Performs a type 1 transition around the edge edge01

    See ../../doc/illus/t1_transition.png for a sketch of the definition
    of the vertices and cells letterings
    """
    # Grab the neighbours
    vert0, vert1, cell_b = sheet.edge_df.loc[
        edge01, ['srce', 'trgt', 'face']].astype(int)

    edge10_ = sheet.edge_df[(sheet.edge_df['srce'] == vert1) &
                        (sheet.edge_df['trgt'] == vert0)]
    edge10 = edge10_.index[0]
    cell_d = int(edge10_.loc[edge10, 'face'])

    edge05_ = sheet.edge_df[(sheet.edge_df['srce'] == vert0) &
                        (sheet.edge_df['face'] == cell_d)]
    edge05 = edge05_.index[0]
    vert5 = int(edge05_.loc[edge05, 'trgt'])

    edge50_ = sheet.edge_df[(sheet.edge_df['srce'] == vert5) &
                        (sheet.edge_df['trgt'] == vert0)]
    edge50 = edge50_.index[0]
    cell_a = int(edge50_.loc[edge50, 'face'])

    edge13_ = sheet.edge_df[(sheet.edge_df['srce'] == vert1) &
                        (sheet.edge_df['face'] == cell_b)]
    edge13 = edge13_.index[0]
    vert3 = int(edge13_.loc[edge13, 'trgt'])

    edge31_ = sheet.edge_df[(sheet.edge_df['srce'] == vert3) &
                        (sheet.edge_df['trgt'] == vert1)]
    edge31 = edge31_.index[0]
    cell_c = int(edge31_.loc[edge31, 'face'])

    edge13_ = sheet.edge_df[(sheet.edge_df['srce'] == vert1) &
                        (sheet.edge_df['face'] == cell_b)]
    edge13 = edge13_.index[0]
    vert3 = int(edge13_.loc[edge13, 'trgt'])

    # Perform the rearangements

    sheet.edge_df.loc[edge01, 'face'] = int(cell_c)
    sheet.edge_df.loc[edge10, 'face'] = int(cell_a)
    sheet.edge_df.loc[edge13, ['srce', 'trgt', 'face']] = vert0, vert3, cell_b
    sheet.edge_df.loc[edge31, ['srce', 'trgt', 'face']] = vert3, vert0, cell_c

    sheet.edge_df.loc[edge50, ['srce', 'trgt', 'face']] = vert5, vert1, cell_a
    sheet.edge_df.loc[edge05, ['srce', 'trgt', 'face']] = vert1, vert5, cell_d

    # Displace the vertices
    mean_pos = (sheet.vert_df.loc[vert0, sheet.coords] +
                sheet.vert_df.loc[vert1, sheet.coords]) / 2
    cell_b_pos = sheet.face_df.loc[cell_b, sheet.coords]
    sheet.vert_df.loc[vert0, sheet.coords] = (mean_pos -
                                          (mean_pos - cell_b_pos) * epsilon)
    cell_d_pos = sheet.face_df.loc[cell_d, sheet.coords]
    sheet.vert_df.loc[vert1, sheet.coords] = (mean_pos -
                                          (mean_pos - cell_d_pos) * epsilon)
    sheet.reset_topo()


def add_vert(sheet, edge):

    srce, trgt = sheet.edge_df.loc[edge, ['srce', 'trgt']]
    opposite = sheet.edge_df[(sheet.edge_df['srce'] == trgt)
                           & (sheet.edge_df['trgt'] == srce)]
    opp_edge = opposite.index[0]

    new_vert = sheet.vert_df.loc[[srce, trgt]].mean()
    sheet.vert_df = sheet.vert_df.append(new_vert, ignore_index=True)
    new_vert = sheet.vert_df.index[-1]
    sheet.edge_df.loc[edge, 'trgt'] = new_vert
    sheet.edge_df.loc[opp_edge, 'srce'] = new_vert

    edge_cols = sheet.edge_df.loc[edge]
    sheet.edge_df = sheet.edge_df.append(edge_cols, ignore_index=True)
    new_edge = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge, 'srce'] = new_vert
    sheet.edge_df.loc[new_edge, 'trgt'] = trgt

    edge_cols = sheet.edge_df.loc[opp_edge]
    sheet.edge_df = sheet.edge_df.append(edge_cols, ignore_index=True)
    new_opp_edge = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_opp_edge, 'trgt'] = new_vert
    sheet.edge_df.loc[new_opp_edge, 'srce'] = trgt
    return new_vert, new_edge, new_opp_edge


def cell_division(sheet, mother, geom, angle=None):

    if not sheet.face_df.loc[mother, 'is_alive']:
        logger.warning('Cell {} is not alive and cannot devide'.format(mother))
        return

    if angle is None:
        angle = np.random.random() * np.pi

    m_data = sheet.edge_df[sheet.edge_df['face'] == mother]
    rot_pos = geom.face_proedgected_pos(sheet, mother, psi=angle)
    srce_pos = rot_pos.loc[m_data['srce'], 'x']
    srce_pos.index = m_data.index
    trgt_pos = rot_pos.loc[m_data['trgt'], 'x']
    trgt_pos.index = m_data.index

    edge_a = m_data[(srce_pos < 0) & (trgt_pos > 0)].index[0]
    edge_b = m_data[(srce_pos > 0) & (trgt_pos < 0)].index[0]


    vert_a, new_edge_a, new_opp_edge_a = add_vert(sheet, edge_a)
    vert_b, new_edge_b, new_opp_edge_b = add_vert(sheet, edge_b)

    face_cols = sheet.face_df.loc[mother]
    sheet.face_df = sheet.face_df.append(face_cols,
                                         ignore_index=True)
    daughter = int(sheet.face_df.index[-1])


    edge_cols = sheet.edge_df.loc[new_edge_b]
    sheet.edge_df = sheet.edge_df.append(edge_cols, ignore_index=True)
    new_edge_m = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge_m, 'srce'] = vert_b
    sheet.edge_df.loc[new_edge_m, 'trgt'] = vert_a

    sheet.edge_df = sheet.edge_df.append(edge_cols, ignore_index=True)
    new_edge_d = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge_d, 'srce'] = vert_a
    sheet.edge_df.loc[new_edge_d, 'trgt'] = vert_b

    daughter_edges = list(m_data[srce_pos < 0].index) + [new_edge_b, new_edge_d]
    sheet.edge_df.loc[daughter_edges, 'face'] = daughter
    sheet.reset_topo()
    geom.update_all(sheet)