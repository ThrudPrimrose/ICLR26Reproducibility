subroutine tsvc_2_s3111_fp64(a, b, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: len_1d
  real(c_double), intent(in) :: a(len_1d)
  real(c_double), intent(out) :: b(2)
  type(c_ptr) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double) :: s1, s2, s3, s4, s5, s6, s7, s8, st
  integer(c_int64_t) :: i, j, n, nb

  n = len_1d
  s1 = 0.0d0; s2 = 0.0d0; s3 = 0.0d0; s4 = 0.0d0
  s5 = 0.0d0; s6 = 0.0d0; s7 = 0.0d0; s8 = 0.0d0
  st = 0.0d0
  nb = (n - 1) / 64
  !$omp parallel do reduction(+:s1,+:s2,+:s3,+:s4,+:s5,+:s6,+:s7,+:s8)
  do i = 1, nb

    do j = 1, 8
      if (a(64*i - 63 + j) > 0.0d0) s1 = s1 + a(64*i - 63 + j)
    end do
    do j = 1, 8
      if (a(64*i - 55 + j) > 0.0d0) s2 = s2 + a(64*i - 55 + j)
    end do
    do j = 1, 8
      if (a(64*i - 47 + j) > 0.0d0) s3 = s3 + a(64*i - 47 + j)
    end do
    do j = 1, 8
      if (a(64*i - 39 + j) > 0.0d0) s4 = s4 + a(64*i - 39 + j)
    end do
    do j = 1, 8
      if (a(64*i - 31 + j) > 0.0d0) s5 = s5 + a(64*i - 31 + j)
    end do
    do j = 1, 8
      if (a(64*i - 23 + j) > 0.0d0) s6 = s6 + a(64*i - 23 + j)
    end do
    do j = 1, 8
      if (a(64*i - 15 + j) > 0.0d0) s7 = s7 + a(64*i - 15 + j)
    end do
    do j = 1, 8
      if (a(64*i - 7 + j) > 0.0d0) s8 = s8 + a(64*i - 7 + j)
    end do
  end do
  do i = 64*nb + 1, n
    if (a(i) > 0.0d0) st = st + a(i)
  end do
  b(1) = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + st
end subroutine tsvc_2_s3111_fp64
