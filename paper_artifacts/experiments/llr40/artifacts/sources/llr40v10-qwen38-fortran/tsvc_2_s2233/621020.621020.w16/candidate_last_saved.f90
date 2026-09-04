subroutine tsvc_2_s2233_fp64(aa, bb, cc, len2d) bind(C, name="tsvc_2_s2233_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), value :: aa, bb, cc
  integer(c_int64_t), value :: len2d
  integer, parameter :: NK = 32
  integer(c_int64_t) :: n, ncol, ng, i0, g, r, m, i0rem
  integer :: kk
  real(c_double), pointer, dimension(:,:) :: A, B, CC2
  real(c_double), dimension(NK) :: acc

  n = len2d
  if (n <= 8) return
  call c_f_pointer(aa, A, [n, n])
  call c_f_pointer(bb, B, [n, n])
  call c_f_pointer(cc, CC2, [n, n])

  ncol = n - 8
  ng = ncol / NK

  ! ---- loop A: for each column i, aa(j,i) = aa(j-1,i) + cc(j,i) ----
  !$omp parallel do schedule(static)
  do g = 0, ng-1
     i0 = 8 + g*NK
     acc = A(8, i0+1 : i0+NK)
     do r = 9, n
        do kk = 1, NK
           acc(kk) = acc(kk) + CC2(r, i0+kk)
           A(r, i0+kk) = acc(kk)
        end do
     end do
  end do
  i0rem = 8 + ng*NK
  if (i0rem < n) then
     m = n - i0rem
     do kk = 1, m
        acc(kk) = A(8, i0rem+kk)
     end do
     do r = 9, n
        do kk = 1, m
           acc(kk) = acc(kk) + CC2(r, i0rem+kk)
           A(r, i0rem+kk) = acc(kk)
        end do
     end do
  end if

  ! ---- loop B: for each row i, bb(i,j) = bb(i-1,j) + cc(i,j) ----
  !$omp parallel do schedule(static)
  do g = 0, ng-1
     i0 = 8 + g*NK
     acc = B(8, i0+1 : i0+NK)
     do r = 9, n
        do kk = 1, NK
           acc(kk) = acc(kk) + CC2(r, i0+kk)
           B(r, i0+kk) = acc(kk)
        end do
     end do
  end do
  i0rem = 8 + ng*NK
  if (i0rem < n) then
     m = n - i0rem
     do kk = 1, m
        acc(kk) = B(8, i0rem+kk)
     end do
     do r = 9, n
        do kk = 1, m
           acc(kk) = acc(kk) + CC2(r, i0rem+kk)
           B(r, i0rem+kk) = acc(kk)
        end do
     end do
  end if
end subroutine tsvc_2_s2233_fp64
