/* eslint-disable */
// this is an auto generated file. This will be overwritten

export const getCo2 = /* GraphQL */ `
  query GetCo2($id: ID!) {
    getCo2(id: $id) {
      id
      name
      value
      timestamp
      anamoly
      
    }
  }
`;
export const listCo2s = /* GraphQL */ `
  query ListCo2s(
    $filter: ModelCo2FilterInput
    $limit: Int
    $nextToken: String
  ) {
    listCo2s(filter: $filter, limit: $limit, nextToken: $nextToken) {
      items {
        id
        name
        value
        timestamp
        anamoly
      }
      nextToken
    }
  }
`;
export const getTemparature = /* GraphQL */ `
  query GetTemparature($id: ID!) {
    getTemparature(id: $id) {
      id
      name
      value
      timestamp
      anamoly
    }
  }
`;
export const listTemparatures = /* GraphQL */ `
  query ListTemparatures(
    $filter: ModelTemparatureFilterInput
    $limit: Int
    $nextToken: String
  ) {
    listTemparatures(filter: $filter, limit: $limit, nextToken: $nextToken) {
      items {
        id
        name
        value
        timestamp
        anamoly
      }
      nextToken
    }
  }
`;
export const getDewpoint = /* GraphQL */ `
  query GetDewpoint($id: ID!) {
    getDewpoint(id: $id) {
      id
      name
      value
      timestamp
      anamoly
    }
  }
`;
export const listDewpoints = /* GraphQL */ `
  query ListDewpoints(
    $filter: ModelDewpointFilterInput
    $limit: Int
    $nextToken: String
  ) {
    listDewpoints(filter: $filter, limit: $limit, nextToken: $nextToken) {
      items {
        id
        name
        value
        timestamp
        anamoly
      }
      nextToken
    }
  }
`;
export const getRelativehumidity = /* GraphQL */ `
  query GetRelativehumidity($id: ID!) {
    getRelativehumidity(id: $id) {
      id
      name
      value
      timestamp
      anamoly
      
    }
  }
`;
export const listRelativehumiditys = /* GraphQL */ `
  query ListRelativehumiditys(
    $filter: ModelRelativehumidityFilterInput
    $limit: Int
    $nextToken: String
  ) {
    listRelativehumiditys(
      filter: $filter
      limit: $limit
      nextToken: $nextToken
    ) {
      items {
        id
        name
        value
        timestamp
        anamoly
      }
      nextToken
    }
  }
`;
