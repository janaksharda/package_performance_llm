import astropy.units as u


def interconnect_epb(interconnect_type, wire_length = None):
    
    if interconnect_type == 'TSV':
        energy = 0.02 * u.pJ / u.bit, 0.005 * u.pJ / u.bit

    if interconnect_type == 'HB':
        energy = 0.01 * u.pJ / u.bit, 0 * u.pJ / u.bit

    if interconnect_type == 'ubump':
        energy = 0.5 * u.pJ / u.bit, 0.1 * u.pJ / u.bit

    if interconnect_type == '2.5D_Si':
        energy = 0.1 * u.pJ / u.bit / u.mm * wire_length, 0.1 * u.pJ / u.bit / u.mm * wire_length

    if interconnect_type == '2.5D_Or':
        energy = 0.2 * u.pJ / u.bit / u.mm * wire_length, 0.05 * u.pJ / u.bit

    if interconnect_type == 'BEOL':
        energy = 0.1 * u.pJ / u.bit / u.mm * wire_length, 0 * u.pJ / u.bit

    return energy

def interconnect_area(interconnect_type, technology_param, wire_length = None):

    if interconnect_type == 'TSV':
        return 5 * u.um ** 2 / u.bit, 1 * u.um ** 2 / u.bit

    if interconnect_type == 'HB':
        return 1 * u.um ** 2 / u.bit, 0.1 * u.um ** 2 / u.bit

    if interconnect_type == 'ubump':
        return 30 * u.um ** 2 / u.bit, 0.1 * u.um ** 2 / u.bit

    if interconnect_type == '2.5D_Si':
        return 1 * u.um ** 2 / u.bit, 0.1 * u.um ** 2 / u.bit

    if interconnect_type == '2.5D_Or':
        return 30 * u.um ** 2 / u.bit, 0.1 * u.um ** 2 / u.bit
        
